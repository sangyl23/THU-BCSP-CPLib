import torch

def customized_block_diag_dft(M, N):
    # generate block diag dft matrix space_vec, (2 * M * N, 2 * M * N)
    B = torch.zeros(M * N, M * N)
    
    m_idx = torch.arange(M)
    n_idx = torch.arange(N)
     
    Fm = torch.exp(1j * 2 * torch.pi / M * m_idx[:, None] * m_idx[None, :])
    Fn = torch.exp(1j * 2 * torch.pi / N * n_idx[:, None] * n_idx[None, :])
     
    B = torch.kron(Fm, Fn)
     
    space_vec = torch.block_diag(B, B) / (M * N) ** 0.5 # (2 * M * N, 2 * M * N)
   
    return space_vec

def customized_dft(RB_num):
    # generate dft matrix delay_vec, (RB_num, RB_num)
    
    idx = torch.arange(RB_num)

    mat = torch.exp(1j * 2 * torch.pi / RB_num * idx[:, None] * idx[None, :])

    delay_vec = mat.conj() / RB_num ** 0.5 # (RB_num, RB_num)
   
    return delay_vec


def channel_to_angle_delay(channel, space_vec, delay_vec, transform_type='IDFT'):
    # transform a channel between spatial-frequency and angle-delay domains
    if transform_type.upper() == 'DFT':
        space_transform = space_vec.conj() # (1, 1, 1, M, M)
        delay_transform = delay_vec # (1, 1, 1, RB_num, RB_num)
    elif transform_type.upper() == 'IDFT':
        space_transform = space_vec # (1, 1, 1, M, M)
        delay_transform = delay_vec.conj() # (1, 1, 1, RB_num, RB_num)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    channel_AD = torch.matmul(space_transform, channel.unsqueeze(dim=-1)).squeeze(dim=-1) # (b, seq_len, RB_num, M)
    channel_AD = channel_AD.permute(0, 1, 3, 2) # (b, seq_len, M, RB_num)
    channel_AD = torch.matmul(channel_AD.unsqueeze(dim=-2), delay_transform).squeeze(dim=-2) # (b, seq_len, M, RB_num)
    channel_AD = channel_AD.permute(0, 1, 3, 2) # (b, seq_len, RB_num, M)
    return channel_AD
