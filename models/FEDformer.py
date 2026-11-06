import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import CSI_VariableEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
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

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        
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

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select,
                                                      num_heads=configs.n_heads)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):
        
        # (B, L, D)
        x = x_enc_AD if self.ifAD else x_enc
        
        # decomp init
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm)
        dec_out = self.dec_embedding(seasonal_init, None, speed_norm, time_interval_norm)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
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
