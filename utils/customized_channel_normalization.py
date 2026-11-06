import torch
from utils.customized_dft_tranformation import channel_to_angle_delay


def _to_complex_channel(channel):
    if torch.is_complex(channel):
        return channel
    if channel.ndim >= 1 and channel.shape[-1] == 2:
        return torch.complex(channel[..., 0], channel[..., 1])
    raise TypeError("Expected a complex channel tensor or a real/imaginary pair")


def _to_real_channel(channel):
    if torch.is_complex(channel):
        return torch.stack([channel.real, channel.imag], dim=-1)
    if channel.ndim >= 1 and channel.shape[-1] == 2:
        return channel
    raise TypeError("Expected a complex channel tensor or a real/imaginary pair")


def channel_preprocess(input_mechanism, instance_norm, batch_x, space_vec=None, delay_vec=None, speed=None, time_interval=None, speed_norm=None, time_interval_norm=None, transform_type='DFT'):
    batch_x_complex = _to_complex_channel(batch_x) # (b, seq_len, RB_num, M)
    normalization_stats = {'mean': None, 'std': None, 'scale': None} # (dict: normalization statistics)
    batch_x_AD = None # (b, seq_len, RB_num, M, 2) or None

    if input_mechanism == 'Channel_independent':
        if instance_norm == 'RevIN':
            normalization_stats['mean'] = batch_x_complex.mean(dim=1, keepdim=True) # (b, 1, RB_num, M)
            normalization_stats['std'] = torch.sqrt(torch.mean(torch.abs(batch_x_complex - normalization_stats['mean']) ** 2, dim=1, keepdim=True)).clamp_min(1e-5) # (b, 1, RB_num, M)
            batch_x_complex = (batch_x_complex - normalization_stats['mean']) / normalization_stats['std'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_2':
            normalization_stats['scale'] = torch.linalg.norm(batch_x_complex, ord=2, dim=1, keepdim=True).clamp_min(1e-5) # (b, 1, RB_num, M)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_max':
            normalization_stats['scale'] = torch.abs(batch_x_complex).max(dim=1, keepdim=True).values.clamp_min(1e-5) # (b, 1, RB_num, M)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm not in ('wo', 'without'):
            raise ValueError(f"Unknown instance norm: {instance_norm}")

        if space_vec is not None and delay_vec is not None:
            batch_x_AD_complex = channel_to_angle_delay(batch_x_complex, space_vec, delay_vec, transform_type=transform_type) # (b, seq_len, RB_num, M)
            batch_x_AD = _to_real_channel(batch_x_AD_complex) # (b, seq_len, RB_num, M, 2)
            batch_x_AD = batch_x_AD.permute(0, 2, 3, 1, 4) # (b, RB_num, M, seq_len, 2)
            batch_x_AD = batch_x_AD.reshape(-1, batch_x_AD.shape[3], 2) # (b * RB_num * M, seq_len, 2)
        
        _, _, RB_num, M, _ = batch_x.shape

        batch_x = _to_real_channel(batch_x_complex) # (b, seq_len, RB_num, M, 2)
        batch_x = batch_x.permute(0, 2, 3, 1, 4) # (b, RB_num, M, seq_len, 2)
        batch_x = batch_x.reshape(-1, batch_x.shape[3], 2) # (b * RB_num * M, seq_len, 2)
        
        # expand condition parameters dim from b -> b * RB_num * M
        if speed is not None:
            speed = speed.repeat_interleave(RB_num * M)
            time_interval = time_interval.repeat_interleave(RB_num * M)
            speed_norm = speed_norm.repeat_interleave(RB_num * M)
            time_interval_norm = time_interval_norm.repeat_interleave(RB_num * M)
        
    elif input_mechanism == 'Channel_mixing':
        if instance_norm == 'RevIN':
            normalization_stats['mean'] = batch_x_complex.mean(dim=(1, 2, 3), keepdim=True) # (b, 1, 1, 1)
            normalization_stats['std'] = torch.sqrt(torch.mean(torch.abs(batch_x_complex - normalization_stats['mean']) ** 2, dim=(1, 2, 3), keepdim=True)).clamp_min(1e-5) # (b, 1, 1, 1)
            batch_x_complex = (batch_x_complex - normalization_stats['mean']) / normalization_stats['std'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_2':
            normalization_stats['scale'] = torch.linalg.vector_norm(batch_x_complex, ord=2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-5) # (b, 1, 1, 1)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_max':
            normalization_stats['scale'] = torch.abs(batch_x_complex).amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5) # (b, 1, 1, 1)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm not in ('wo'):
            raise ValueError(f"Unknown instance norm: {instance_norm}")

        if space_vec is not None and delay_vec is not None:
            batch_x_AD_complex = channel_to_angle_delay(batch_x_complex, space_vec, delay_vec, transform_type=transform_type) # (b, seq_len, RB_num, M)
            batch_x_AD = _to_real_channel(batch_x_AD_complex) # (b, seq_len, RB_num, M, 2)
            batch_x_AD = batch_x_AD.reshape(batch_x_AD.shape[0], batch_x_AD.shape[1], -1) # (b, seq_len, RB_num * M * 2)

        batch_x = _to_real_channel(batch_x_complex) # (b, seq_len, RB_num, M, 2)
        batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], -1) # (b, seq_len, RB_num * M * 2)

    elif input_mechanism == 'RB_independent':
        _, _, RB_num, M = batch_x_complex.shape

        if instance_norm == 'RevIN':
            normalization_stats['mean'] = batch_x_complex.mean(dim=(1, 3), keepdim=True) # (b, 1, RB_num, 1)
            normalization_stats['std'] = torch.sqrt(torch.mean(torch.abs(batch_x_complex - normalization_stats['mean']) ** 2, dim=(1, 3), keepdim=True)).clamp_min(1e-5) # (b, 1, RB_num, 1)
            batch_x_complex = (batch_x_complex - normalization_stats['mean']) / normalization_stats['std'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_2':
            normalization_stats['scale'] = torch.linalg.vector_norm(batch_x_complex, ord=2, dim=(1, 3), keepdim=True).clamp_min(1e-5) # (b, 1, RB_num, 1)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_max':
            normalization_stats['scale'] = torch.abs(batch_x_complex).amax(dim=(1, 3), keepdim=True).clamp_min(1e-5) # (b, 1, RB_num, 1)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm not in ('wo', 'without'):
            raise ValueError(f"Unknown instance norm: {instance_norm}")

        if space_vec is not None and delay_vec is not None:
            batch_x_AD_complex = channel_to_angle_delay(batch_x_complex, space_vec, delay_vec, transform_type=transform_type) # (b, seq_len, RB_num, M)
            batch_x_AD = _to_real_channel(batch_x_AD_complex) # (b, seq_len, RB_num, M, 2)
            batch_x_AD = batch_x_AD.permute(0, 2, 1, 3, 4) # (b, RB_num, seq_len, M, 2)
            batch_x_AD = batch_x_AD.reshape(-1, batch_x_AD.shape[2], M * 2) # (b * RB_num, seq_len, M * 2)

        batch_x = _to_real_channel(batch_x_complex) # (b, seq_len, RB_num, M, 2)
        batch_x = batch_x.permute(0, 2, 1, 3, 4) # (b, RB_num, seq_len, M, 2)
        batch_x = batch_x.reshape(-1, batch_x.shape[2], M * 2) # (b * RB_num, seq_len, M * 2)

        # expand condition parameters dim from b -> b * RB_num
        if speed is not None:
            speed = speed.repeat_interleave(RB_num)
            time_interval = time_interval.repeat_interleave(RB_num)
            speed_norm = speed_norm.repeat_interleave(RB_num)
            time_interval_norm = time_interval_norm.repeat_interleave(RB_num)
    
    elif input_mechanism == 'M_independent':
        _, _, RB_num, M = batch_x_complex.shape

        if instance_norm == 'RevIN':
            normalization_stats['mean'] = batch_x_complex.mean(dim=(1, 2), keepdim=True) # (b, 1, 1, M)
            normalization_stats['std'] = torch.sqrt(torch.mean(torch.abs(batch_x_complex - normalization_stats['mean']) ** 2, dim=(1, 2), keepdim=True)).clamp_min(1e-5) # (b, 1, 1, M)
            batch_x_complex = (batch_x_complex - normalization_stats['mean']) / normalization_stats['std'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_2':
            normalization_stats['scale'] = torch.linalg.vector_norm(batch_x_complex, ord=2, dim=(1, 2), keepdim=True).clamp_min(1e-5) # (b, 1, 1, M)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm == 'norm_max':
            normalization_stats['scale'] = torch.abs(batch_x_complex).amax(dim=(1, 2), keepdim=True).clamp_min(1e-5) # (b, 1, 1, M)
            batch_x_complex = batch_x_complex / normalization_stats['scale'] # (b, seq_len, RB_num, M)
        elif instance_norm not in ('wo', 'without'):
            raise ValueError(f"Unknown instance norm: {instance_norm}")

        if space_vec is not None and delay_vec is not None:
            batch_x_AD_complex = channel_to_angle_delay(batch_x_complex, space_vec, delay_vec, transform_type=transform_type) # (b, seq_len, RB_num, M)
            batch_x_AD = _to_real_channel(batch_x_AD_complex) # (b, seq_len, RB_num, M, 2)
            batch_x_AD = batch_x_AD.permute(0, 3, 1, 2, 4) # (b, M, seq_len, RB_num, 2)
            batch_x_AD = batch_x_AD.reshape(-1, batch_x_AD.shape[2], RB_num * 2) # (b * M, seq_len, RB_num * 2)

        batch_x = _to_real_channel(batch_x_complex) # (b, seq_len, RB_num, M, 2)
        batch_x = batch_x.permute(0, 3, 1, 2, 4) # (b, M, seq_len, RB_num, 2)
        batch_x = batch_x.reshape(-1, batch_x.shape[2], RB_num * 2) # (b * M, seq_len, RB_num * 2)

        # expand condition parameters dim from b -> b * M
        if speed is not None:
            speed = speed.repeat_interleave(M)
            time_interval = time_interval.repeat_interleave(M)
            speed_norm = speed_norm.repeat_interleave(M)
            time_interval_norm = time_interval_norm.repeat_interleave(M)

    else:
        raise ValueError(f"Unknown input mechanism: {input_mechanism}")

    return batch_x, batch_x_AD, normalization_stats, speed, time_interval, speed_norm, time_interval_norm


def channel_postprocess(input_mechanism, instance_norm, out_tensor, normalization_stats, RB_num, M, pred_len, ifAD=False, space_vec=None, delay_vec=None):
    if input_mechanism == 'Channel_independent':
        out_tensor = out_tensor.reshape(-1, RB_num, M, pred_len, 2) # (b, RB_num, M, pred_len, 2)
        out_tensor = out_tensor.permute(0, 3, 1, 2, 4) # (b, pred_len, RB_num, M, 2)
    elif input_mechanism == 'Channel_mixing':
        out_tensor = out_tensor.reshape(-1, pred_len, RB_num, M, 2) # (b, pred_len, RB_num, M, 2)
    elif input_mechanism == 'RB_independent':
        out_tensor = out_tensor.reshape(-1, RB_num, pred_len, M, 2) # (b, RB_num, pred_len, M, 2)
        out_tensor = out_tensor.permute(0, 2, 1, 3, 4) # (b, pred_len, RB_num, M, 2)
    elif input_mechanism == 'M_independent':
        out_tensor = out_tensor.reshape(-1, M, pred_len, RB_num, 2) # (b, M, pred_len, RB_num, 2)
        out_tensor = out_tensor.permute(0, 2, 3, 1, 4) # (b, pred_len, RB_num, M, 2)
    else:
        raise ValueError(f"Unknown input mechanism: {input_mechanism}")

    out_tensor_complex = _to_complex_channel(out_tensor) # (b, pred_len, RB_num, M)
    if ifAD:
        if space_vec is None or delay_vec is None:
            raise ValueError("space_vec and delay_vec are required for IDFT postprocessing")
        out_tensor_complex = channel_to_angle_delay(out_tensor_complex, space_vec, delay_vec, transform_type='IDFT') # (b, pred_len, RB_num, M)
        
    if instance_norm == 'RevIN':
        out_tensor_complex = out_tensor_complex * normalization_stats['std'] + normalization_stats['mean'] # (b, pred_len, RB_num, M)
    elif instance_norm in ('norm_2', 'norm_max'):
        out_tensor_complex = out_tensor_complex * normalization_stats['scale'] # (b, pred_len, RB_num, M)
    elif instance_norm not in ('wo'):
        raise ValueError(f"Unknown instance norm: {instance_norm}")

    out_tensor = _to_real_channel(out_tensor_complex) # (b, pred_len, RB_num, M, 2)
    return out_tensor
