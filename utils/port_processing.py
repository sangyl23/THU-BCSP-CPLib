import torch


def port_preprocess(channel, mode='full', rb_within_group=None, m_within_group=None,
                    uplink=None, group_num=None):
    """Apply the common port-selection operation before normalization."""
    if mode == 'full':
        return channel
    if rb_within_group is None or m_within_group is None:
        raise ValueError('Grouped port modes require group dimensions.')
    complex_channel = torch.complex(channel[..., 0], channel[..., 1])
    rb_num, antenna_num = complex_channel.shape[-2:]
    if group_num is None:
        group_num = (rb_num * antenna_num) // (rb_within_group * m_within_group)
    if mode == 'full_long_period':
        time_mask = torch.arange(complex_channel.shape[1], device=channel.device) % group_num == 0
        complex_channel = complex_channel * time_mask.view(1, -1, 1, 1)
        return torch.stack((complex_channel.real, complex_channel.imag), dim=-1)
    if mode == 'partial_downsample':
        mask = torch.zeros((rb_num, antenna_num), dtype=torch.bool, device=channel.device)
        mask[::rb_num // rb_within_group, ::antenna_num // m_within_group] = True
        complex_channel = complex_channel * mask
        return torch.stack((complex_channel.real, complex_channel.imag), dim=-1)
    if mode == 'partial_strong':
        if uplink is None:
            raise ValueError('partial_strong requires uplink CSI.')
        magnitudes = torch.abs(torch.complex(uplink[..., 0], uplink[..., 1]))
        topk = rb_within_group * m_within_group
        _, indices = torch.topk(magnitudes.reshape(*magnitudes.shape[:-2], -1), topk, dim=-1)
        mask = torch.zeros_like(magnitudes.reshape(*magnitudes.shape[:-2], -1))
        selected = (complex_channel.reshape(*complex_channel.shape[:-2], -1)
                    * mask.scatter(-1, indices, 1)).reshape_as(complex_channel)
        return torch.stack((selected.real, selected.imag), dim=-1)
    if mode in ('grouped_cyclic_block', 'grouped_cyclic_angle',
                'grouped_cyclic_angle_delay',
                'full_long_period_split_downsample_short_period'):
        flat_ports = torch.arange(rb_num * antenna_num, device=channel.device)
        port_groups = flat_ports % group_num
        time_groups = torch.arange(complex_channel.shape[1], device=channel.device) % group_num
        mask = port_groups.view(1, 1, rb_num, antenna_num) == time_groups.view(1, -1, 1, 1)
        complex_channel = complex_channel * mask
        return torch.stack((complex_channel.real, complex_channel.imag), dim=-1)
    if mode == 'full_long_period_fixed_short_period':
        return channel
    if mode != 'full':
        raise ValueError(f'Unknown port estimation mode: {mode}')
    return channel


def port_postprocess(prediction, input_mechanism, rb_num, antenna_num, pred_len):
    """Restore model output to (batch, pred_len, RB, M, 2)."""
    if input_mechanism == 'Channel_independent':
        prediction = prediction.reshape(-1, rb_num, antenna_num, pred_len, 2)
        return prediction.permute(0, 3, 1, 2, 4)
    return prediction.reshape(-1, pred_len, rb_num, antenna_num, 2)
