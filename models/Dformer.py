import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
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
        self.ifvis = getattr(configs, 'ifvis', False)
        self.saved_path = getattr(configs, 'saved_path', 'tmp/')

        self.emb_module = configs.emb_module # define how to conduct token and positional embedding
        self.embed_learning = configs.embed_learning # define learning in doppler domain or time domain
        self.doppler_before_embedding = getattr(
            configs, 'doppler_before_embedding', False
        )
        if self.doppler_before_embedding and self.embed_learning != 'doppler':
            raise ValueError(
                'doppler_before_embedding=True requires embed_learning="doppler".'
            )
        self.doppler_pre_embedding = (
            self.embed_learning == 'doppler' and self.doppler_before_embedding
        )
        self.doppler_keep_ratio = configs.doppler_keep_ratio
        num_speed_conditions = getattr(
            configs, 'num_speed_conditions', getattr(configs, 'num_conditions', 4)
        )
        num_interval_conditions = getattr(
            configs, 'num_interval_conditions', getattr(configs, 'num_conditions', 4)
        )
        
        # Doppler before embedding truncates the original temporal dimension;
        # Doppler after embedding truncates the embedded feature dimension.
        if self.doppler_pre_embedding:
            if self.emb_module != 'temporal_emb':
                raise ValueError(
                    'Doppler-before-embedding currently requires temporal_emb.'
                )
            self.cutoff = max(
                1, min(self.seq_len, int(self.seq_len * configs.doppler_keep_ratio))
            )
        elif self.emb_module == 'temporal_emb':
            self.cutoff = int(configs.d_model * configs.doppler_keep_ratio)
        else:
            self.cutoff = int(self.seq_len * configs.doppler_keep_ratio)
        #######################################################################
        
        #######################################################################
        # embedding module       
        if self.emb_module == 'temporal_emb':
            # Independent-CSI variable embedding: (B, seq_len, D) -> (B, D, d_model).
            # Before-embedding Doppler changes its temporal input to cutoff.
            embedding_length = (
                self.cutoff if self.doppler_pre_embedding else self.seq_len
            )
            self.enc_embedding = CSI_TimeEmbedding(
                embedding_length, configs.d_model, configs.embed, 'h',
                configs.dropout, configs.ifemb_conditions,
                configs.condition_embedding, num_speed_conditions,
                num_interval_conditions,
                merge_complex=self.doppler_pre_embedding
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
        if configs.embed_learning == 'doppler' and not self.doppler_pre_embedding:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False), self.cutoff, configs.n_heads),
                        self.cutoff,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.cutoff)
            )
        else:
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
        #######################################################################
        
        #######################################################################
        # decoder module
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':              
            if self.emb_module == 'temporal_emb':
                # Before-embedding Doppler predicts real/imaginary components
                # for every target-frequency coefficient: d_model -> 2 * pred_len.
                projection_length = (
                    2 * self.pred_len
                    if self.doppler_pre_embedding else self.pred_len
                )
                self.projection = nn.Linear(
                    configs.d_model, projection_length, bias=True
                )
            else:
                self.projection_var = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.projection_tem = nn.Linear(self.seq_len, self.pred_len, bias=True)               
        #######################################################################

    def vis(self, **stages):
        """Save complete (B, L, D) complex tensors and plot B=0, D=0.

        Each call creates a separate directory so successive batches do not
        overwrite previous results. Close all six windows to resume execution.
        """
        import os
        import tempfile
        import matplotlib.pyplot as plt
        from scipy.io import savemat

        os.makedirs(self.saved_path, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix='doppler_vis_', dir=self.saved_path)
        for name, tensor in stages.items():
            data = tensor.detach().cpu().numpy()
            savemat(os.path.join(output_dir, name + '.mat'), {name: data})
            trajectory = data[0, :, 0]
            fig, ax = plt.subplots(num=name)
            ax.plot(trajectory.real, label='Real')
            ax.plot(trajectory.imag, label='Imaginary')
            ax.plot(np.abs(trajectory), label='Magnitude', linestyle='--')
            ax.set_title(name + ' (B=0, D=0)')
            ax.set_xlabel('Time index' if name in (
                'fft_input', 'ifft_output'
            ) else 'Doppler bin (FFT order; cutoff bins concatenated)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        print('Doppler visualization saved to: ' + os.path.abspath(output_dir))
        plt.show(block=True)

    def forecast(self, x_enc, x_enc_AD, x_mark_enc, 
                 x_dec, x_dec_AD, x_mark_dec, 
                 speed, time_interval, 
                 speed_norm, time_interval_norm,
                 t_idx_enc, t_idx_dec):

        # (B, L, D)
        x = x_enc_AD if self.ifAD else x_enc

        # Doppler transform before embedding: retain low-frequency coefficients
        # from the original complex CSI sequence.
        if self.doppler_pre_embedding:
            batch_size, sequence_length, input_size = x.shape
            if input_size % 2 != 0:
                raise ValueError(
                    'Doppler-before-embedding requires adjacent real/imaginary pairs.'
                )
            complex_history = torch.complex(x[..., 0::2], x[..., 1::2])
            # (B, seq_len, 2 * variable_num) -> (B, seq_len, variable_num)
            spectrum = torch.fft.fft(complex_history, norm='ortho', dim=1)
            # spectrum: (B, seq_len, variable_num)
            front = (self.cutoff + 1) // 2
            back = self.cutoff // 2
            frequency_indices = torch.cat([
                torch.arange(front, device=x.device),
                torch.arange(sequence_length - back, sequence_length, device=x.device),
            ])
            spectrum = spectrum[:, frequency_indices, :] # (B, cutoff, variable_num)
            x = torch.stack([spectrum.real, spectrum.imag], dim=-1).reshape(
                batch_size, self.cutoff, input_size
            ) # (B, cutoff, 2 * variable_num)

        # Embedding
        if self.emb_module == 'temporal_emb':
            # merge_complex=True in the before-embedding Doppler branch:
            # (B, cutoff, 2 * variable_num) -> (B, variable_num, d_model).
            enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm) # (B, D, L)
        else:
            enc_out = self.enc_embedding(x, None, speed_norm, time_interval_norm) # (B, L, D)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Learning in doppler domain or time domain
        # if learning in doppler domain, we need to conduct fft and high-frequency cutoff
        if self.embed_learning == 'doppler' and not self.doppler_pre_embedding:
            enc_out = enc_out.permute(0, 2, 1) # (B, L, D)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1, 2)
            enc_out_complex = torch.complex(enc_out[..., 0], enc_out[..., 1])
            enc_out = torch.fft.fft(enc_out_complex, norm = "ortho", dim = 1) # (B, L, D)
            if self.ifvis:
                vis_stages = dict(fft_input=enc_out_complex, fft_output=enc_out)
                                  
            front = (self.cutoff + 1) // 2
            back = self.cutoff // 2
        
            idx = torch.cat([
                torch.arange(front, device = enc_out.device),
                torch.arange(enc_out.shape[1] - back, enc_out.shape[1], device = enc_out.device)
            ])
            
            original_freq = torch.zeros_like(enc_out)
            enc_out = enc_out[:, idx, :]
            if self.ifvis:
                vis_stages['cutoff_output'] = enc_out
            
            enc_out = torch.stack([enc_out.real, enc_out.imag], dim = -1)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Encoder learning time and variable correlations        
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # (B, D, L)

        # if learning in doppler domain, we need to conduct high-frequency zerofilling and ifft
        if self.embed_learning == 'doppler' and not self.doppler_pre_embedding:
            enc_out = enc_out.permute(0, 2, 1) # (B, L, D)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1, 2)
            enc_out = torch.complex(enc_out[..., 0], enc_out[..., 1])
            if self.ifvis:
                vis_stages['encoder_output'] = enc_out
            
            original_freq[:, idx, :] = enc_out
            
            enc_out_complex = torch.fft.ifft(original_freq, norm = "ortho", dim = 1)
            if self.ifvis:
                self.vis(
                    **vis_stages,
                    zerofill_output=original_freq,
                    ifft_output=enc_out_complex,
                )
            enc_out = torch.stack([enc_out_complex.real, enc_out_complex.imag], dim = -1)
            enc_out = enc_out.reshape(enc_out.shape[0], enc_out.shape[1], -1)
            enc_out = enc_out.permute(0, 2, 1) # (B, D, L)

        # Decoder project output
        if self.emb_module == 'temporal_emb':
            if self.doppler_pre_embedding:
                # Predict the target Doppler spectrum for every variable token.
                dec_out = self.projection(enc_out) # (B, variable_num, 2 * pred_len)
                dec_out = dec_out.reshape(
                    batch_size, input_size // 2, self.pred_len, 2
                ) # (B, variable_num, pred_len, 2)
                dec_out = torch.complex(dec_out[..., 0], dec_out[..., 1])
                # (B, variable_num, pred_len), complex-valued target spectrum.
                dec_out = torch.fft.ifft(dec_out, norm='ortho', dim=2)
                # Inverse Doppler transform: frequency domain -> target time domain.
                dec_out = torch.stack(
                    [dec_out.real, dec_out.imag], dim=-1
                ).permute(0, 2, 1, 3).reshape(
                    batch_size, self.pred_len, input_size
                ) # (B, pred_len, 2 * variable_num)
            else:
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
