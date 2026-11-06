import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import CSI_VariableEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
      
        #######################################################################
        ## parameters setup
        self.task_name = configs.task_name
        self.ifAD = configs.ifAD
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        num_speed_conditions = getattr(
            configs, 'num_speed_conditions', getattr(configs, 'num_conditions', 4)
        )
        num_interval_conditions = getattr(
            configs, 'num_interval_conditions', getattr(configs, 'num_conditions', 4)
        )
        #######################################################################
        
        #######################################################################
        # Embedding
        self.enc_embedding = CSI_VariableEmbedding(configs.enc_in, configs.d_model, 
                                                   configs.dropout,
                                                   configs.ifemb_conditions,
                                                   configs.condition_embedding,
                                                   'fixed', num_speed_conditions,
                                                   num_interval_conditions)
        self.dec_embedding = CSI_VariableEmbedding(configs.dec_in, configs.d_model, 
                                                   configs.dropout,
                                                   configs.ifemb_conditions,
                                                   configs.condition_embedding,
                                                   'fixed', num_speed_conditions,
                                                   num_interval_conditions)
        #######################################################################
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )


    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):
        
        # (B, L, D)
        x1 = x_enc_AD if self.ifAD else x_enc
        x2 = x_dec_AD if self.ifAD else x_dec
                
        # embedding
        enc_out = self.enc_embedding(x1, None, speed_norm, time_interval_norm)
        dec_out = self.dec_embedding(x2, None, speed_norm, time_interval_norm)
        
        # enc
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # dec
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
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
