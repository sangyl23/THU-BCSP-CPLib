import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import CSI_TimeEmbedding
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
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
        
        # Embedding
        self.enc_embedding = CSI_TimeEmbedding(
            configs.seq_len, configs.d_model, configs.embed, 'h',
            configs.dropout, configs.ifemb_conditions,
            configs.condition_embedding, num_speed_conditions,
            num_interval_conditions,
            merge_complex = False
        )
        
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
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):
        
        # (B, L, D)
        x = x_enc_AD if self.ifAD else x_enc
        
        _, _, N = x.shape

        # Embedding
        enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

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
