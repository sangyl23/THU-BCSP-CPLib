import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class CSI_VariableEmbedding_PE(nn.Module):
    """Generate condition-aware sinusoidal encoding for CSI time samples."""

    def __init__(self, d_model):
        """Initialize the encoding frequencies for an even embedding dimension."""
        super(CSI_VariableEmbedding_PE, self).__init__()
        if d_model % 2 != 0:
            raise ValueError('CSI_VariableEmbedding requires an even d_model.')
        self.d_model = d_model
        # Sinusoidal frequency scales, (d_model / 2,).
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, speed_norm, time_interval_norm, L):
        """
        Encode physical sampling positions with speed-dependent frequencies.

        Args:
            speed_norm: Min-normalized speed, (B,).
            time_interval_norm: Min-normalized sampling interval, (B,).
            L: Input sequence length.

        Returns:
            Condition-aware sinusoidal encoding, (B, L, d_model).
        """
        if speed_norm is None or time_interval_norm is None:
            raise ValueError(
                'frequency_pe requires speed_norm and time_interval_norm.'
            )
        if L < 1:
            raise ValueError('L must be positive.')

        speed_norm = speed_norm.reshape(-1)  # Min-normalized speed, (B,).
        time_interval_norm = time_interval_norm.reshape(-1)  # Min-normalized interval, (B,).
        if speed_norm.shape != time_interval_norm.shape:
            raise ValueError(
                'speed_norm and time_interval_norm must have the same shape.'
            )

        device = speed_norm.device
        dtype = speed_norm.dtype if speed_norm.is_floating_point() else torch.float32
        speed_norm = speed_norm.to(dtype=dtype)
        time_interval_norm = time_interval_norm.to(device=device, dtype=dtype)
        # Base sample indices shared by the batch, (1, L, 1).
        position = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)
        position = position * time_interval_norm.view(-1, 1, 1)  # Physical positions, (B, L, 1).
        # Speed-modulated sinusoidal phases, (B, L, d_model / 2).
        phase = (
            speed_norm.view(-1, 1, 1)
            * position
            * self.div_term.to(device=device, dtype=dtype).view(1, 1, -1)
        )

        pe = torch.zeros(  # Condition-aware position encoding, (B, L, d_model).
            speed_norm.shape[0], L, self.d_model, device=device, dtype=dtype
        )
        pe[..., 0::2] = torch.sin(phase)
        pe[..., 1::2] = torch.cos(phase)
        return pe


class CSI_VariableEmbedding_Conditions(nn.Module):
    """Embed discrete speed and sampling-interval conditions separately."""

    def __init__(self, d_model, num_speed_conditions=4,
                 num_interval_conditions=4, embed_type='fixed'):
        """Initialize two condition lookup tables with d_model-dimensional outputs."""
        super(CSI_VariableEmbedding_Conditions, self).__init__()
        embedding = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        # Speed-condition lookup table: category index -> (d_model,).
        self.speed_embedding = embedding(num_speed_conditions, d_model)
        # Sampling-interval lookup table: category index -> (d_model,).
        self.interval_embedding = embedding(num_interval_conditions, d_model)
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions

    @staticmethod
    def _to_index(value, num_conditions):
        """
        Convert normalized condition values to embedding indices.

        Args:
            value: Min-normalized values in {1, ..., C}, or values in [0, 1], (B,).
            num_conditions: Number of discrete condition categories.

        Returns:
            Clamped zero-based embedding indices, (B,).
        """
        value = value.reshape(-1)  # Normalized condition values, (B,).
        if value.is_floating_point():
            value_min = value.detach().amin()
            if value_min < 1.0 - torch.finfo(value.dtype).eps:
                index = torch.round(value * (num_conditions - 1)).long()  # Indices, (B,).
                return index.clamp(0, num_conditions - 1)
        index = torch.round(value).long() - 1  # Zero-based indices, (B,).
        return index.clamp(0, num_conditions - 1)

    def forward(self, speed_norm, time_interval_norm, L):
        """
        Add speed and interval embeddings and broadcast them over time.

        Args:
            speed_norm: Normalized speed category, (B,).
            time_interval_norm: Normalized interval category, (B,).
            L: Input sequence length.

        Returns:
            Per-sample condition embedding, (B, L, d_model).
        """
        if speed_norm is None or time_interval_norm is None:
            raise ValueError(
                'separate requires speed_norm and time_interval_norm.'
            )
        speed_index = self._to_index(
            speed_norm, self.num_speed_conditions
        )  # Speed indices, (B,).
        # Sampling-interval indices, (B,).
        interval_index = self._to_index(
            time_interval_norm, self.num_interval_conditions
        )
        embedding_device = next(self.speed_embedding.parameters()).device
        speed_index = speed_index.to(device=embedding_device)
        interval_index = interval_index.to(device=embedding_device)
        # Combined speed and sampling-interval embedding, (B, d_model).
        condition = (
            self.speed_embedding(speed_index)
            + self.interval_embedding(interval_index)
        )
        return condition.unsqueeze(1).expand(-1, L, -1)


class CSI_VariableEmbedding(nn.Module):
    """Embed mixed CSI variables with a selectable condition encoding."""

    def __init__(self, c_in, d_model, dropout=0.1,
                 ifemb_conditions=True, condition_embedding='frequency_pe',
                 embed_type='fixed', num_speed_conditions=4,
                 num_interval_conditions=4):
        """Initialize CSI value, position, and selectable condition embeddings."""
        super(CSI_VariableEmbedding, self).__init__()

        if condition_embedding not in ('frequency_pe', 'continuous', 'separate', 'concat'):
            raise ValueError(
                "condition_embedding must be 'frequency_pe', 'continuous', "
                "'separate', or 'concat'."
            )

        value_channels = c_in + 2 if ifemb_conditions and condition_embedding == 'concat' else c_in
        self.value_embedding = TokenEmbedding(c_in=value_channels, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.ifemb_conditions = ifemb_conditions
        self.condition_embedding_type = condition_embedding
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions

        if ifemb_conditions and condition_embedding != 'concat':
            if condition_embedding == 'frequency_pe':
                self.condition_embedding = CSI_VariableEmbedding_PE(d_model=d_model)
            elif condition_embedding == 'continuous':
                self.condition_embedding = CSI_TimeEmbedding_Continuous(
                    d_model=d_model,
                    num_speed_conditions=num_speed_conditions,
                    num_interval_conditions=num_interval_conditions,
                )
            else:
                self.condition_embedding = CSI_VariableEmbedding_Conditions(
                    d_model=d_model,
                    num_speed_conditions=num_speed_conditions,
                    num_interval_conditions=num_interval_conditions,
                    embed_type=embed_type,
                )
        else:
            self.condition_embedding = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, speed_norm, time_interval_norm):
        """
        Embed a CSI sequence together with speed and interval conditions.

        Args:
            x: CSI input sequence, (B, L, c_in).
            x_mark: Reserved temporal markers; unused in this embedding.
            speed_norm: Min-normalized speed, (B,).
            time_interval_norm: Min-normalized sampling interval, (B,).

        Returns:
            Embedded CSI sequence, (B, L, d_model).
        """
        L = x.shape[1]
        if self.ifemb_conditions and self.condition_embedding_type == 'concat':
            conditions = _scale_csi_conditions(
                speed_norm, time_interval_norm,
                self.num_speed_conditions, self.num_interval_conditions
            )
            if conditions.shape[0] != x.shape[0]:
                raise ValueError('Condition batch size must match the CSI batch size.')
            conditions = conditions.to(device=x.device, dtype=x.dtype)
            conditions = conditions.unsqueeze(1).expand(-1, L, -1)
            x_value = self.value_embedding(torch.cat([x, conditions], dim=-1))
        else:
            x_value = self.value_embedding(x)  # CSI value embedding, (B, L, d_model).

        if self.ifemb_conditions:
            if self.condition_embedding_type == 'frequency_pe':
                # Condition-aware sinusoidal encoding, (B, L, d_model).
                x_condition = self.condition_embedding(
                    speed_norm, time_interval_norm, L
                )
                x = x_value + x_condition  # Embedded CSI sequence, (B, L, d_model).
            elif self.condition_embedding_type == 'continuous':
                x_condition = self.condition_embedding(
                    speed_norm, time_interval_norm
                ).unsqueeze(1)
                x = x_value + x_condition
            elif self.condition_embedding_type == 'separate':
                # Broadcast discrete condition embedding, (B, L, d_model).
                x_condition = self.condition_embedding(
                    speed_norm, time_interval_norm, L
                )
                x = x_value + self.position_embedding(x) + x_condition  # (B, L, d_model).
            else:
                x = x_value + self.position_embedding(x)
        else:
            x = x_value + self.position_embedding(x)

        return self.dropout(x)


def _scale_csi_conditions(speed_norm, time_interval_norm,
                          num_speed_conditions, num_interval_conditions):
    """Map min-normalized CSI conditions from [1, C] to [-1, 1]."""
    if speed_norm is None or time_interval_norm is None:
        raise ValueError(
            'Condition embedding requires speed_norm and time_interval_norm.'
        )
    speed_norm = speed_norm.reshape(-1)
    time_interval_norm = time_interval_norm.reshape(-1)
    if speed_norm.shape != time_interval_norm.shape:
        raise ValueError(
            'speed_norm and time_interval_norm must have the same shape.'
        )

    dtype = speed_norm.dtype if speed_norm.is_floating_point() else torch.float32
    speed_norm = speed_norm.to(dtype=dtype)
    time_interval_norm = time_interval_norm.to(
        device=speed_norm.device, dtype=dtype
    )
    speed_denominator = max(num_speed_conditions - 1, 1)
    interval_denominator = max(num_interval_conditions - 1, 1)
    speed_scaled = 2.0 * (speed_norm - 1.0) / speed_denominator - 1.0
    interval_scaled = (
        2.0 * (time_interval_norm - 1.0) / interval_denominator - 1.0
    )
    return torch.stack([speed_scaled, interval_scaled], dim=-1)


class CSI_TimeEmbedding_PE(nn.Module):
    """Encode two continuous CSI conditions with bounded Fourier features."""

    def __init__(self, d_model, num_speed_conditions=4,
                 num_interval_conditions=4):
        super(CSI_TimeEmbedding_PE, self).__init__()
        if d_model % 4 != 0:
            raise ValueError(
                'frequency_pe for CSI_TimeEmbedding requires d_model divisible by 4.'
            )
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions
        condition_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, condition_dim, 2).float()
            * -(math.log(10000.0) / condition_dim)
        )
        self.register_buffer('div_term', div_term)

    def _fourier_encode(self, value, num_conditions):
        phase = (
            0.5
            * (value.unsqueeze(-1) + 1.0)
            * (num_conditions - 1)
            * self.div_term.to(device=value.device, dtype=value.dtype)
        )
        encoding = torch.empty(
            value.shape[0], self.div_term.numel() * 2,
            device=value.device, dtype=value.dtype
        )
        encoding[:, 0::2] = torch.sin(phase)
        encoding[:, 1::2] = torch.cos(phase)
        return encoding

    def forward(self, speed_norm, time_interval_norm):
        conditions = _scale_csi_conditions(
            speed_norm, time_interval_norm,
            self.num_speed_conditions, self.num_interval_conditions
        )
        return torch.cat([
            self._fourier_encode(conditions[:, 0], self.num_speed_conditions),
            self._fourier_encode(
                conditions[:, 1], self.num_interval_conditions
            ),
        ], dim=-1)


class CSI_TimeEmbedding_Continuous(nn.Module):
    """Learn a smooth additive embedding for two continuous CSI conditions."""

    def __init__(self, d_model, num_speed_conditions=4,
                 num_interval_conditions=4):
        super(CSI_TimeEmbedding_Continuous, self).__init__()
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions
        self.projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, speed_norm, time_interval_norm):
        conditions = _scale_csi_conditions(
            speed_norm, time_interval_norm,
            self.num_speed_conditions, self.num_interval_conditions
        )
        conditions = conditions.to(
            device=self.projection[0].weight.device,
            dtype=self.projection[0].weight.dtype,
        )
        return self.projection(conditions)


class CSI_TimeEmbedding_Conditions(nn.Module):
    """Embed discrete conditions without losing their speed/interval roles."""

    def __init__(self, d_model, num_speed_conditions=4,
                 num_interval_conditions=4, embed_type='fixed'):
        super(CSI_TimeEmbedding_Conditions, self).__init__()
        embedding = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.speed_embedding = embedding(num_speed_conditions, d_model)
        self.interval_embedding = embedding(num_interval_conditions, d_model)
        self.projection = nn.Linear(2 * d_model, d_model, bias=False)
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions

    def forward(self, speed_norm, time_interval_norm, num_tokens):
        speed_index = CSI_VariableEmbedding_Conditions._to_index(
            speed_norm, self.num_speed_conditions
        )
        interval_index = CSI_VariableEmbedding_Conditions._to_index(
            time_interval_norm, self.num_interval_conditions
        )
        embedding_device = next(self.speed_embedding.parameters()).device
        speed_index = speed_index.to(device=embedding_device)
        interval_index = interval_index.to(device=embedding_device)
        condition = self.projection(torch.cat([
            self.speed_embedding(speed_index),
            self.interval_embedding(interval_index),
        ], dim=-1))
        return condition.unsqueeze(1).expand(-1, num_tokens, -1)


class CSI_ChannelConditionModulation(nn.Module):
    """Generate channel-specific affine modulation from global conditions."""

    def __init__(self, condition_dim, d_model):
        super(CSI_ChannelConditionModulation, self).__init__()
        self.channel_embedding = PositionalEmbedding(d_model=d_model)
        self.modulation = nn.Sequential(
            nn.Linear(condition_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, x, condition, condition_scale):
        if condition.ndim != 2:
            raise ValueError('condition must have shape (B, condition_dim).')
        if condition.shape[0] != x.shape[0]:
            raise ValueError('Condition batch size must match the CSI batch size.')

        condition = condition.to(device=x.device, dtype=x.dtype)
        condition = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
        channel = self.channel_embedding(x).to(device=x.device, dtype=x.dtype)
        channel = channel.expand(x.shape[0], -1, -1)
        gamma, beta = self.modulation(
            torch.cat([condition, channel], dim=-1)
        ).chunk(2, dim=-1)
        scale = condition_scale.to(device=x.device, dtype=x.dtype)
        return x * (1.0 + scale * torch.tanh(gamma)) + scale * torch.tanh(beta)


class CSI_TimeEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 ifemb_conditions=True, condition_embedding='frequency_pe',
                 num_speed_conditions=4, num_interval_conditions=4,
                 merge_complex=False):
        super(CSI_TimeEmbedding, self).__init__()
        valid_embeddings = ('frequency_pe', 'continuous', 'separate', 'concat', 'frequency_pe_wo_flim')
        if condition_embedding not in valid_embeddings:
            raise ValueError(
                f'condition_embedding must be one of {valid_embeddings}.'
            )

        self.merge_complex = merge_complex
        value_input_dim = 2 * c_in if merge_complex else c_in
        self.value_embedding = nn.Linear(value_input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.ifemb_conditions = ifemb_conditions
        self.condition_embedding_type = condition_embedding
        self.num_speed_conditions = num_speed_conditions
        self.num_interval_conditions = num_interval_conditions
        self.register_buffer(
            'condition_scale', torch.tensor(1.0), persistent=False
        )

        if not ifemb_conditions or condition_embedding == 'concat':
            self.condition_embedding = None
        elif condition_embedding == 'frequency_pe' or condition_embedding == 'frequency_pe_wo_flim':
            self.condition_embedding = CSI_TimeEmbedding_PE(
                d_model=d_model,
                num_speed_conditions=num_speed_conditions,
                num_interval_conditions=num_interval_conditions,
            )
        elif condition_embedding == 'continuous':
            self.condition_embedding = CSI_TimeEmbedding_Continuous(
                d_model=d_model,
                num_speed_conditions=num_speed_conditions,
                num_interval_conditions=num_interval_conditions,
            )
        else:
            self.condition_embedding = CSI_TimeEmbedding_Conditions(
                d_model=d_model,
                num_speed_conditions=num_speed_conditions,
                num_interval_conditions=num_interval_conditions,
                embed_type=embed_type,
            )
        condition_dim = 2 if condition_embedding == 'concat' else d_model
        self.condition_modulation = (
            CSI_ChannelConditionModulation(condition_dim, d_model)
            if (ifemb_conditions and condition_embedding != 'frequency_pe_wo_flim') else None
        )

    def forward(self, x, x_mark, speed_norm, time_interval_norm):
        """
        Embed a CSI sequence together with speed and interval conditions.

        Args:
            x: CSI input sequence, (B, seq_len, D). When merge_complex=True,
                adjacent entries in D must be real/imaginary pairs.
            x_mark: Reserved temporal markers; unused in this embedding.
            speed_norm: Min-normalized speed category in [1, C], (B,).
            time_interval_norm: Min-normalized interval category in [1, C], (B,).

        Returns:
            Embedded CSI sequence, (B, D, d_model), or (B, D / 2, d_model)
            when merge_complex=True.
        """
        
        if self.merge_complex:
            if x.shape[-1] % 2 != 0:
                raise ValueError(
                    'Complex CSI embedding requires adjacent real/imaginary pairs.'
                )
            x = x.reshape(x.shape[0], x.shape[1], -1, 2)
            x = x.permute(0, 2, 1, 3).reshape(x.shape[0], -1, 2 * x.shape[1])
        else:
            x = x.permute(0, 2, 1)

        x = self.value_embedding(x)
        if self.ifemb_conditions:
            if self.condition_embedding_type == 'concat':
                condition = _scale_csi_conditions(
                    speed_norm, time_interval_norm,
                    self.num_speed_conditions, self.num_interval_conditions
                )
            elif self.condition_embedding_type == 'separate':
                condition = self.condition_embedding(
                    speed_norm, time_interval_norm, 1
                ).squeeze(dim = 1)
            else: # frequency_pe or continuous
                condition = self.condition_embedding(
                    speed_norm, time_interval_norm
                )  
            
            if self.condition_embedding_type == 'frequency_pe_wo_flim':
                x = x + condition.unsqueeze(1)
            else:
                # flim modulation
                x = self.condition_modulation(
                    x, condition, self.condition_scale
                )

        # x: [Batch Variate d_model]
        return self.dropout(x)
