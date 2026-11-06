import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec_ffn_attn import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import CSI_TimeEmbedding, CSI_VariableEmbedding
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        #######################################################################
        ## parameters setup
        self.task_name = configs.task_name
        self.ifAD = configs.ifAD
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.emb_module = configs.emb_module # define how to conduct token and positional embedding
        self.embed_learning = configs.embed_learning # define learning in doppler domain or time domain
        self.doppler_keep_ratio = configs.doppler_keep_ratio
        num_speed_conditions = getattr(
            configs, 'num_speed_conditions', getattr(configs, 'num_conditions', 4)
        )
        num_interval_conditions = getattr(
            configs, 'num_interval_conditions', getattr(configs, 'num_conditions', 4)
        )
        
        if self.emb_module == 'temporal_emb':
            self.cutoff = int(configs.d_model * configs.doppler_keep_ratio)
        else:
            self.cutoff = int(self.seq_len * configs.doppler_keep_ratio)
        #######################################################################
        
        #######################################################################
        # embedding module       
        if self.emb_module == 'temporal_emb':
            # Independent-CSI variable embedding: (B, seq_len, D) -> (B, D, d_model).
            self.enc_embedding = CSI_TimeEmbedding(
                self.seq_len, configs.d_model, configs.embed, 'h',
                configs.dropout, configs.ifemb_conditions,
                configs.condition_embedding, num_speed_conditions,
                num_interval_conditions
            )
        else:
            # Mixed-CSI variable embedding: (B, L, enc_in) -> (B, L, d_model).
            self.enc_embedding = CSI_VariableEmbedding(configs.enc_in, configs.d_model, 
                                                       configs.dropout,
                                                       configs.ifemb_conditions,
                                                       configs.condition_embedding,
                                                       'fixed', num_speed_conditions,
                                                       num_interval_conditions)
        #######################################################################
        
        #######################################################################
        # encoder module
        # learning in doppler domain or time domain
        if configs.embed_learning == 'doppler':
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False), 2 * configs.RB_num * configs.M, configs.n_heads),
                        2 * configs.RB_num * configs.M,
                        2 * configs.RB_num * configs.M,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.cutoff)
            )
        else:
            raise ValueError('Ablation just consider doppler learning.')
        #######################################################################
        
        #######################################################################
        # decoder module
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':              
            if self.emb_module == 'temporal_emb':
                self.projection = nn.Linear(configs.d_model, self.pred_len, bias=True)
            else:
                self.projection_var = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.projection_tem = nn.Linear(self.seq_len, self.pred_len, bias=True)               
        #######################################################################

    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):

        # (B, L, D)
        x = x_enc_AD if self.ifAD else x_enc

        # Embedding
        if self.emb_module == 'temporal_emb':
            enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm) # (B, D, L)
        else:
            enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm) # (B, L, D)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Learning in doppler domain or time domain
        # if learning in doppler domain, we need to conduct fft and high-frequency cutoff
        if self.embed_learning == 'doppler':
            enc_out = enc_out.permute(0, 2, 1) # (B, L, D)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1, 2)
            enc_out_complex = torch.complex(enc_out[..., 0], enc_out[..., 1])
            enc_out = torch.fft.fft(enc_out_complex, norm = "ortho", dim = 1) # (B, L, D)
                                  
            front = (self.cutoff + 1) // 2
            back = self.cutoff // 2
        
            idx = torch.cat([
                torch.arange(front, device = enc_out.device),
                torch.arange(enc_out.shape[1] - back, enc_out.shape[1], device = enc_out.device)
            ])
            
            original_freq = torch.zeros_like(enc_out)
            enc_out = enc_out[:, idx, :]
            
            enc_out = torch.stack([enc_out.real, enc_out.imag], dim = -1)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Encoder learning time and variable correlations        
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, D, L)

        # if learning in doppler domain, we need to conduct high-frequency zerofilling and ifft
        if self.embed_learning == 'doppler':
            enc_out = enc_out.permute(0, 2, 1) # (B, L, D)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1, 2)
            enc_out = torch.complex(enc_out[..., 0], enc_out[..., 1])
            
            original_freq[:, idx, :] = enc_out
            
            enc_out_complex = torch.fft.ifft(original_freq, norm = "ortho", dim = 1)
            enc_out = torch.stack([enc_out_complex.real, enc_out_complex.imag], dim = -1)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Decoder project output
        if self.emb_module == 'temporal_emb':
            dec_out = self.projection(enc_out).permute(0, 2, 1) # (B, L, D)
        else:
            dec_out = self.projection_var(enc_out.permute(0, 2, 1)) # (B, L, D)
            dec_out = self.projection_tem(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, D)
                       
        return dec_out

    def forward(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, mask = None, 
                 speed = None, time_interval = None, 
                 speed_norm = None, time_interval_norm = None,
                 t_idx_enc = None, t_idx_dec = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc=x_enc,
                x_enc_AD=x_enc_AD,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_dec_AD=x_dec_AD,
                x_mark_dec=x_mark_dec,
                speed=speed,
                time_interval=time_interval,
                speed_norm=speed_norm,
                time_interval_norm=time_interval_norm,
                t_idx_enc = t_idx_enc,
                t_idx_dec = t_idx_dec
            )
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
