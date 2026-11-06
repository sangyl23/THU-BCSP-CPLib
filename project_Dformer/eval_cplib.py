import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
for import_path in (PROJECT_ROOT, PROJECT_DIR):
    import_path = str(import_path)
    if import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

import torch
import numpy as np
import scipy.io as sio
from dataloader_cplib import Dataloader
from utils.customized_dft_tranformation import customized_block_diag_dft, customized_dft
from utils.customized_channel_normalization import channel_postprocess, channel_preprocess
from utils.model_profile import print_model_statistics
import time
import random
import math
import argparse
import logging
import re
# from thop import profile

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument('--model_name', type=str, default = 'Dformer',
                          choices=['LSTM', 
                                   'Transformer',
                                   'Linformer',
                                   'FEDformer',
                                   'TimesNet',
                                   'PatchTST',
                                   'iTransformer',
                                   'Dformer',
                                   'Dformer_ffn_ffn',
                                   'Dformer_attn_attn',
                                   'Dformer_ffn_attn'])
known_args, remaining_argv = model_parser.parse_known_args()
model_name = known_args.model_name

parser = argparse.ArgumentParser(parents=[model_parser])

from configs.channel_system_parameters_setup import add_channel_system_args
parser = add_channel_system_args(parser)

if model_name == 'LSTM':
    from models.LSTM import Model
    from configs.lstm_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Transformer':
    from models.Transformer import Model
    from configs.transformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Linformer':
    from models.Linformer import Model
    from configs.linformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'FEDformer':
    from models.FEDformer import Model
    from configs.fedformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'TimesNet':
    from models.TimesNet import Model
    from configs.timesnet_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'PatchTST':
    from models.PatchTST import Model
    from configs.patchtst_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'iTransformer':
    from models.iTransformer import Model
    from configs.itransformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Dformer':
    from models.Dformer import Model
    from configs.dformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Dformer_ffn_ffn':
    from models.Dformer_ffn_ffn import Model
    from configs.dformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Dformer_attn_attn':
    from models.Dformer_attn_attn import Model
    from configs.dformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
elif model_name == 'Dformer_ffn_attn':
    from models.Dformer_ffn_attn import Model
    from configs.dformer_parameters_setup import add_model_args
    parser = add_model_args(parser)
else:
    raise ValueError(f"Unknown model name: {model_name}")      
    
from configs.training_parameters_setup import add_training_args
parser = add_training_args(parser)

parser.add_argument('--checkpoint_name', type = str, default = 'xx')

args = parser.parse_args()

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(args.saved_path + "logfile.txt", mode = 'w'),
    logging.StreamHandler()
])
logger = logging.getLogger()


def print_model_structure(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + f"{name}: {module.__class__.__name__}")
        print_model_structure(module, indent + 2)

## calculate NMSE
def channel_NMSE(h_gt, h_pre):
    # (b * UE_num, pred_len, RB_num, M, 2)
    h_gt = torch.complex(h_gt[:, :, :, :, 0], h_gt[:, :, :, :, 1]) # (b * UE_num, pred_len, RB_num, M)
    h_pre = torch.complex(h_pre[:, :, :, :, 0], h_pre[:, :, :, :, 1]) # (b * UE_num, pred_len, RB_num, M)

    return torch.mean(torch.linalg.norm((h_gt - h_pre), dim = (3)) ** 2 / 
                      torch.linalg.norm(h_gt, dim = (3)) ** 2, dim = (0, 2)), torch.mean(torch.linalg.norm(h_pre, dim = (3)), dim = (0, 2)) # (pred_len,)

## calculate SGCS
def channel_SGCS(h_gt, h_recover, b, UE_num, pred_len, RB_num, M):
    # (b * UE_num, pred_len, RB_num, M, 2)
    h_gt = torch.complex(h_gt[:, :, :, :, 0], h_gt[:, :, :, :, 1]) # (b * UE_num, pred_len, RB_num, M)
    h_gt = h_gt.permute(0, 2, 1, 3) # (b * UE_num, RB_num, pred_len, M)
    h_gt = h_gt.reshape(-1, pred_len, M) # (b * UE_num * RB_num, pred_len, M)
    h_recover = torch.complex(h_recover[:, :, :, :, 0], h_recover[:, :, :, :, 1]) # (b * UE_num, pred_len, RB_num, M)
    h_recover = h_recover.permute(0, 2, 1, 3) # (b * UE_num, RB_num, pred_len, M)
    h_recover = h_recover.reshape(-1, pred_len, M) # (b * UE_num * RB_num, pred_len, M)
    
    numerator = torch.sum(h_gt.conj() * h_recover, dim = 2).abs() ** 2
    denominator = (torch.linalg.norm(h_gt, dim = 2) ** 2) * (torch.linalg.norm(h_recover, dim = 2) ** 2)
    sgcs = torch.mean(numerator / denominator, dim = 0)  # (pred_len,)

    return sgcs

## calculate spectral efficiency
def MultiUE_SE(h_gt, h_recover, b, UE_num, pred_len, RB_num, M):
    # (b * UE_num, pred_len, RB_num, M, 2)
    h_gt = h_gt.reshape(b, UE_num, pred_len, RB_num, M, 2)
    h_gt = h_gt.permute(0, 2, 3, 1, 5, 4) # (b, pred_len, RB_num, UE_num, 2, M)
    h_gt = h_gt.reshape(-1, UE_num, 2, M) # (b * pred_len * RB_num, UE_num, 2, M)
    h_recover = h_recover.reshape(b, UE_num, pred_len, RB_num, M, 2)
    h_recover = h_recover.permute(0, 2, 3, 1, 5, 4) # (b, pred_len, RB_num, UE_num, 2, M)
    h_recover = h_recover.reshape(-1, UE_num, 2, M) # (b * pred_len * RB_num, UE_num, 2, M)

    match = re.search(r'(\d+(?:\.\d+)?)dB', args.snr)
    DLsnr = float(match.group(1))
    power = 1.
    noise = power * 10 ** (- DLsnr / 10)
    
    ## norm_2 for h_gt
    h_gt_complex = torch.complex(torch.squeeze(h_gt[:, :, 0, :]), torch.squeeze(h_gt[:, :, 1, :])) # (b * pred_len * RB_num, UE_num, M)           
    h_gt = h_gt_complex / torch.linalg.norm(h_gt_complex, ord = 2, dim = -1, keepdim = True) # (b * pred_len * RB_num, UE_num, M)
    h_gt_real = h_gt.real # (b * pred_len * RB_num, UE_num, M)
    h_gt_imag = h_gt.imag # (b * pred_len * RB_num, UE_num, M)
    h_gt = torch.stack([h_gt_real, h_gt_imag], dim = 2) # (b * pred_len * RB_num, UE_num, 2, M)  
    
    h_gt = torch.complex(h_gt[:, :, 0, :], h_gt[:, :, 1, :]).squeeze() # (b * pred_len * RB_num, UE_num, M)
    h_recover = torch.complex(h_recover[:, :, 0, :], h_recover[:, :, 1, :]).squeeze() # (b * pred_len * RB_num, UE_num, M)  
    
    # Zero forcing precoding
    W_ZF = torch.linalg.pinv(h_recover) # (b * pred_len * RB_num, M, UE_num)
    W_ZF = W_ZF.permute(0, 2, 1) # (b * pred_len * RB_num, UE_num, M)
    
    # power allocation
    W_ZF = W_ZF / torch.linalg.norm(W_ZF, ord = 2, dim = -1, keepdim = True) # (b * pred_len * RB_num, UE_num, M)
    
    W_ZF = W_ZF.permute(0, 2, 1) * math.sqrt(power) # (b * pred_len * RB_num, M, UE_num)
    
    BS_H_UE = torch.matmul(h_gt, W_ZF) # (b * pred_len * RB_num, UE_num, UE_num)

    BS_H_UE = torch.abs(BS_H_UE)
    HV = BS_H_UE ** 2
    eff_power = torch.diagonal(HV, offset = 0, dim1 = -2, dim2 = -1) # effective power, (b * pred_len * RB_num, UE_num)
    sum_power = torch.sum(HV, dim = 2) # sum power, (b * pred_len * RB_num, UE_num)

    rate_k = (sum_power + noise) / (sum_power - eff_power + noise) # (b * pred_len * RB_num, UE_num)
    rate_k = torch.log2(rate_k) # (b * pred_len * RB_num, UE_num)

    se = torch.sum(rate_k, dim = 1) # (b * pred_len * RB_num)
    se = se.reshape(b, pred_len, RB_num) # (b, pred_len, RB_num)
        
    return torch.mean(se, dim = (0, 2)) # (pred_len,)

# evaluation function
def eval(model, loader, UE_num, RB_num, M, seq_len, pred_len, RB_within_group, M_within_group, group_num, b, device, criterion, space_vec, delay_vec):
    # reset dataloader
    # loader.reset()
    # judge whether dataset is finished
    done = False
    # running loss
    running_NMSE = torch.zeros([pred_len, 1])
    running_pre_norm = torch.zeros([pred_len, 1])
    running_sgcs = torch.zeros([pred_len, 1])
    running_se = torch.zeros([pred_len, 1])
    # count batch number
    batch_num = 0
    
    pre_nodes = torch.zeros([b * UE_num, pred_len, RB_num, M, 2])
    gt_nodes = torch.zeros([b * UE_num, pred_len, RB_num, M, 2])
    
    with torch.no_grad():
        # evaluate validation set
        while True:
            ## read files     
            # his_channel_DL: historical downlink channel sequence, (b, UE_num, seq_len, RB_num, M, 2)                     
            # his_channel_UL: historical uplink channel sequence, (b, UE_num, seq_len, RB_num, M, 2) 
            # pre_channel_DL: future downlink channel sequence, (b, UE_num, pred_len, RB_num, M, 2)  
            # pre_channel_UL: future uplink channel sequence, (b, UE_num, pred_len, RB_num, M, 2)  
            if args.read_uplink:
                his_channel_DL, his_channel_UL, pre_channel_DL, pre_channel_UL, done = loader.next_batch()
                if done:  # 数据集结束
                    break
                his_channel_UL = his_channel_UL.reshape(-1, seq_len, RB_num, M, 2) # (b * UE_num, seq_len, RB_num, M, 2)
            else:
                his_channel_DL, pre_channel_DL, done = loader.next_batch()
                if done:
                    break

            batch_num += 1
            
            ## input preprocess and reshape
            his_channel_DL = his_channel_DL.reshape(-1, seq_len, RB_num, M, 2) # (b * UE_num, seq_len, RB_num, M, 2)
            pre_channel_DL = pre_channel_DL.reshape(-1, pred_len, RB_num, M, 2) # (b * UE_num, pred_len, RB_num, M, 2)
            
            if args.port_estimation_mode == 'full':
                # The loader provides angle-delay channels. Normalize them and
                # transform a second copy to the spatial-frequency domain.
                his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                    args.input_mechanism,
                    args.instance_norm,
                    his_channel_DL,
                    space_vec,
                    delay_vec,
                    None,
                    None,
                    None,
                    None,
                    transform_type='IDFT'
                )
                
                t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float() # (seq_len)
                    
            elif args.port_estimation_mode == 'full_long_period':
                his_channel_DL = his_channel_DL[:, : : group_num, :, :, :] # (b * UE_num, seq_len // group_num, RB_num, M, 2)
                his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                    args.input_mechanism, args.instance_norm, his_channel_DL,
                    space_vec,
                    delay_vec,
                    None,
                    None,
                    None,
                    None,
                    transform_type='IDFT')
                
                t_idx_enc = torch.arange(start = 0, end = seq_len, step = group_num, device = device, requires_grad = False).float() # (seq_len // group_num)
                
            elif args.port_estimation_mode == 'partial_strong':
                his_channel_DL_complex = torch.complex(his_channel_DL[..., 0], his_channel_DL[..., 1]) # (b * UE_num, seq_len, RB_num, M)
                his_channel_UL_complex = torch.complex(his_channel_UL[..., 0], his_channel_UL[..., 1]) # (b * UE_num, seq_len, RB_num, M)
                # Restore the UE axis and use the first UE to select shared downlink pilot ports.
                his_channel_UL_complex = his_channel_UL_complex.reshape(b, UE_num, seq_len, RB_num, M)[:, 0]
                
                magnitudes = his_channel_UL_complex.abs()
            
                # Select the strongest RB-port pairs independently at each time step.
                magnitudes_flat = magnitudes.reshape(b, seq_len, RB_num * M)
                
                topk_num = RB_within_group * M_within_group
                
                _, topk_idx = torch.topk(magnitudes_flat, k = topk_num, dim = -1)
            
                mask_flat = torch.zeros_like(magnitudes_flat, dtype = torch.float32)
                mask_flat.scatter_(dim = -1, index = topk_idx, value = 1.0)
            
                # Share each (b, seq_len, RB_num, M) mask across all UEs in the same sample.
                mask = mask_flat.reshape(b, 1, seq_len, RB_num, M).expand(-1, UE_num, -1, -1, -1)
                mask = mask.reshape(b * UE_num, seq_len, RB_num, M)
            
                his_channel_DL_complex = his_channel_DL_complex * mask
                
                his_channel_DL = torch.stack([his_channel_DL_complex.real, his_channel_DL_complex.imag], dim = -1)
                his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                    args.input_mechanism, args.instance_norm, his_channel_DL,
                    space_vec,
                    delay_vec,
                    None,
                    None,
                    None,
                    None,
                    transform_type='IDFT')
                
                t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float() # (seq_len)
            
            elif args.port_estimation_mode == 'partial_downsample':    
                his_channel_DL_complex = torch.complex(his_channel_DL[..., 0], his_channel_DL[..., 1]) # (b * UE_num, seq_len, RB_num, M)
                mask = torch.zeros_like(his_channel_DL_complex)
                mask[:, :, :: RB_num // RB_within_group, :: M // M_within_group] = 1
                his_channel_DL_complex = his_channel_DL_complex * mask # (b * UE_num, seq_len, RB_num, M)
                his_channel_DL = torch.stack([his_channel_DL_complex.real, his_channel_DL_complex.imag], dim = -1) # (b * UE_num, seq_len, RB_num, M, 2)
                his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                    args.input_mechanism, args.instance_norm, his_channel_DL,
                    space_vec,
                    delay_vec,
                    None,
                    None,
                    None,
                    None,
                    transform_type='IDFT')
                
                t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float() # (seq_len)
                    
            elif args.port_estimation_mode == 'grouped_cyclic_block':
                his_channel_DL_complex = torch.complex(his_channel_DL[:, :, :, :, 0], his_channel_DL[:, :, :, :, 1]).squeeze() # (b * UE_num, seq_len, RB_num, M)
                tmp = his_channel_DL_complex.view(b * UE_num, seq_len, RB_num // RB_within_group, RB_within_group, M // M_within_group, M_within_group) 
                tmp = tmp.permute(0, 1, 3, 5, 2, 4) # (b * UE_num, seq_len, RB_within_group, M_within_group, RB_num // RB_within_group, M // M_within_group) 
                his_channel_DL_complex = tmp.reshape(b * UE_num, seq_len, RB_within_group, M_within_group, group_num) # (b * UE_num, seq_len, RB_within_group, M_within_group, group_num)
                his_channel_DL_real = his_channel_DL_complex.real
                his_channel_DL_imag = his_channel_DL_complex.imag
                his_channel_DL = torch.stack([his_channel_DL_real, his_channel_DL_imag], dim = -1) # (b * UE_num, seq_len, RB_within_group, M_within_group, group_num, 2)
                
                ## preprocess
                if args.input_mechanism == 'Channel_independent':
                    
                    # Fill zero
                    h_idx = torch.arange(seq_len).unsqueeze(1) # (seq_len, 1) 
                    g_idx = torch.arange(group_num).unsqueeze(0) # (1, group_num) 
                    mask_matrix = (g_idx == (h_idx % group_num)).to(his_channel_DL.dtype).to(his_channel_DL.device) # (seq_len, group_num) 
                    mask = mask_matrix.view(1, seq_len, 1, 1, group_num, 1)
                    his_channel_DL = his_channel_DL * mask # (b * UE_num, seq_len, RB_within_group, M_within_group, group_num, 2)
                    
                    his_channel_DL = his_channel_DL.reshape(b * UE_num, seq_len, RB_within_group, M_within_group, RB_num // RB_within_group, M // M_within_group, 2)
                    his_channel_DL = his_channel_DL.permute(0, 1, 4, 2, 5, 3, 6) # (b * UE_num, seq_len, RB_num // RB_within_group, RB_within_group, M // M_within_group, M_within_group, 2)
                    his_channel_DL = his_channel_DL.contiguous().view(b * UE_num, seq_len, RB_num, M, 2)
                    
                    his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                        args.input_mechanism, args.instance_norm, his_channel_DL,
                        space_vec,
                        delay_vec,
                        None,
                        None,
                        None,
                        None,
                        transform_type='IDFT')
                    
                    t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float() # (seq_len)
                                        
                elif args.input_mechanism == 'Channel_mixing':
                    
                    # Fill zero
                    h_idx = torch.arange(seq_len).unsqueeze(1) # (seq_len, 1) 
                    g_idx = torch.arange(group_num).unsqueeze(0) # (1, group_num) 
                    mask_matrix = (g_idx == (h_idx % group_num)).to(his_channel_DL.dtype).to(his_channel_DL.device) # (seq_len, group_num) 
                    mask = mask_matrix.view(1, seq_len, 1, 1, group_num, 1)
                    his_channel_DL = his_channel_DL * mask # (b * UE_num, seq_len, RB_within_group, M_within_group, group_num, 2)
                    
                    his_channel_DL = his_channel_DL.reshape(b * UE_num, seq_len, RB_within_group, M_within_group, RB_num // RB_within_group, M // M_within_group, 2)
                    his_channel_DL = his_channel_DL.permute(0, 1, 4, 2, 5, 3, 6) # (b * UE_num, seq_len, RB_num // RB_within_group, RB_within_group, M // M_within_group, M_within_group, 2)
                    his_channel_DL = his_channel_DL.contiguous().view(b * UE_num, seq_len, RB_num, M, 2)
                    
                    his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                        args.input_mechanism, args.instance_norm, his_channel_DL,
                        space_vec,
                        delay_vec,
                        None,
                        None,
                        None,
                        None,
                        transform_type='IDFT')
                    
                    t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float() # (seq_len)
                    
                else:
                    ValueError(f"Unknown input mechanism: {args.input_mechanism}")
            
            elif args.port_estimation_mode == 'grouped_cyclic_angle_delay':
            
                # his_channel_DL, shape (b * UE_num, seq_len, RB_num, M, 2)
                
                # Step1: convert to complex tensor
                his_channel_DL_complex = torch.complex(his_channel_DL[...,0], his_channel_DL[...,1])  # (b * UE_num, seq_len, RB_num, M)
                
                # Step2: split along RB dimension
                rb_slices = torch.stack(
                    [his_channel_DL_complex[:, :, i::RB_num // RB_within_group, :] for i in range(RB_num // RB_within_group)],
                    dim=2
                )  # (b * UE_num, seq_len, RB_num//RB_within_group, RB_within_group, M)
                
                # Step3: split along M dimension
                m_slices = torch.stack(
                    [rb_slices[:, :, :, :, j::M // M_within_group] for j in range(M // M_within_group)],
                    dim=4
                )  # (b * UE_num, seq_len, RB_num//RB_within_group, RB_within_group, M//M_within_group, M_within_group)
                
                # Step4: permute to group RB_within_group and M_within_group together
                m_slices = m_slices.permute(0,1,3,5,2,4)  
                # (b * UE_num, seq_len, RB_within_group, M_within_group, RB_num//RB_within_group, M//M_within_group)
                
                # Step5: reshape to group_num
                his_channel_DL_complex = m_slices.reshape(b * UE_num, seq_len, RB_within_group, M_within_group, group_num)
                his_channel_DL = torch.stack([his_channel_DL_complex.real, his_channel_DL_complex.imag], dim=-1)
                # (b * UE_num, seq_len, RB_within_group, M_within_group, group_num, 2)
                
                # Step6: build Z-scan mapping (RB priority)
                z_scan_map = []
                for m in range(M // M_within_group):
                    if m % 2 == 0:  # forward scan along RB
                        for rb in range(RB_num // RB_within_group):
                            z_scan_map.append((rb, m))
                    else:  # backward scan along RB
                        for rb in reversed(range(RB_num // RB_within_group)):
                            z_scan_map.append((rb, m))
                z_scan_map = torch.tensor(z_scan_map, device=his_channel_DL.device)  # (group_num, 2)
                
                # Step7: assign indices for each seq_len
                h_idx = torch.arange(seq_len, device=his_channel_DL.device)
                rb_idx = z_scan_map[h_idx % group_num, 0]
                m_idx  = z_scan_map[h_idx % group_num, 1]
                
                # Step8: build mask
                mask = torch.zeros((seq_len, RB_num // RB_within_group, M // M_within_group),
                                   dtype=his_channel_DL.dtype, device=his_channel_DL.device)
                for t in range(seq_len):
                    mask[t, rb_idx[t], m_idx[t]] = 1.0
                
                mask = mask.view(1, seq_len, 1, 1, group_num, 1)
                
                # Step9: apply mask
                his_channel_DL = his_channel_DL * mask
                
                # Step10: reconstruct back to original shape
                his_channel_DL_complex = torch.complex(his_channel_DL[...,0], his_channel_DL[...,1]).squeeze()
                his_channel_DL_complex = his_channel_DL_complex.reshape(
                    b * UE_num, seq_len, RB_within_group, M_within_group, RB_num // RB_within_group, M // M_within_group
                )
                his_channel_DL_complex = his_channel_DL_complex.permute(0,1,2,4,3,5).contiguous().view(b * UE_num, seq_len, RB_num, M)
                
                # Step11: split real/imag and flatten
                his_channel_DL_real = his_channel_DL_complex.real
                his_channel_DL_imag = his_channel_DL_complex.imag
                his_channel_DL = torch.stack([his_channel_DL_real, his_channel_DL_imag], dim=-1)  # (b * UE_num, seq_len, RB_num, M, 2)

                his_channel_DL, his_channel_DL_SF, normalization_stats, _, _, _, _ = channel_preprocess(
                    args.input_mechanism, args.instance_norm, his_channel_DL,
                    space_vec,
                    delay_vec,
                    None,
                    None,
                    None,
                    None,
                    transform_type='IDFT')
                t_idx_enc = torch.arange(seq_len, device = device, requires_grad = False).float()

                
            ## eval the network
            B, L, D = his_channel_DL.shape
            x_enc, x_enc_AD = his_channel_DL_SF, his_channel_DL # (B, L, D)
            x_dec, x_dec_AD = torch.zeros(B, pred_len, D).to(device), torch.zeros(B, pred_len, D).to(device)
            L_label = args.label_len // group_num if args.port_estimation_mode == 'full_long_period' else args.label_len
            x_dec, x_dec_AD =  torch.cat([x_enc[:, L - L_label : L, :], x_dec], dim = 1), torch.cat([x_enc_AD[:, L - L_label : L, :], x_dec_AD], dim = 1)
            
            t_idx_dec = torch.arange(start = seq_len, end = seq_len + pred_len, device = device, requires_grad = False).float() # (pred_len)         
            # t_idx_enc: (seq_len)
            t_idx_dec = torch.cat([t_idx_enc[L - L_label : L], t_idx_dec], dim = 0)
            
            out_tensor = model(x_enc = x_enc, x_enc_AD = x_enc_AD, x_mark_enc = None, 
                                     x_dec = x_dec, x_dec_AD = x_dec_AD, x_mark_dec = None,
                                     mask = None, 
                                     speed = None, time_interval = None, 
                                     speed_norm = None, time_interval_norm = None,
                                     t_idx_enc = t_idx_enc, t_idx_dec = t_idx_dec)

            if batch_num == 1:
                # Print parameter counts and single-sample FLOPs on the first test batch.
                print_model_statistics(
                    model,
                    (None, x_enc, None, None, x_dec, None, None,
                     None, None, None, None, t_idx_enc, t_idx_dec),
                    b,
                )
                              
            ## postprocess
            out_tensor = channel_postprocess(
                args.input_mechanism,
                args.instance_norm,
                out_tensor,
                normalization_stats,
                RB_num,
                M,
                pred_len,
                ifAD=False, # ifAD is set to False, because our objective pre_channel_DL is in AD domain not SF domain
                space_vec=space_vec,
                delay_vec=delay_vec
                )
                
                          
            ## calculate KPIs
            nmse_loss, pre_norm = channel_NMSE(h_gt = pre_channel_DL, h_pre = out_tensor) # (pred_len,)
            
            if batch_num == 1:
                pre_nodes = out_tensor.data.cpu()
                gt_nodes = pre_channel_DL.data.cpu()
            else:
                tmp = out_tensor.data.cpu()
                pre_nodes = torch.cat([pre_nodes, tmp], dim = 0)    
                tmp = pre_channel_DL.data.cpu()
                gt_nodes = torch.cat([gt_nodes, tmp], dim = 0)  
            
            if args.evaluate_sgcs == True or args.evaluate_se == True:
                # angle-delay domain -> spatial-frequency domain
                pre_channel_DL_sf = torch.complex(pre_channel_DL[:, :, :, :, 0], pre_channel_DL[:, :, :, :, 1]).squeeze() # (b * UE_num, pred_len, RB_num, M)
                pre_channel_DL_sf = pre_channel_DL_sf.unsqueeze(dim = -1) # (b * UE_num, pred_len, RB_num, M, 1)
                pre_channel_DL_sf = torch.matmul(space_vec, pre_channel_DL_sf).squeeze(dim = -1) # (b * UE_num, pred_len, RB_num, M)
                pre_channel_DL_sf = pre_channel_DL_sf.permute(0, 1, 3, 2) # (b * UE_num, pred_len, M, RB_num)
                pre_channel_DL_sf = pre_channel_DL_sf.unsqueeze(dim = -2) # (b * UE_num, pred_len, M, 1, RB_num)
                pre_channel_DL_sf = torch.matmul(pre_channel_DL_sf, delay_vec.conj()).squeeze(dim = -2) # (b * UE_num, pred_len, M, RB_num)
                pre_channel_DL_sf = pre_channel_DL_sf.permute(0, 1, 3, 2) # (b * UE_num, pred_len, RB_num, M)
                pre_channel_DL_sf_real = pre_channel_DL_sf.real
                pre_channel_DL_sf_imag = pre_channel_DL_sf.imag
                pre_channel_DL_sf = torch.stack([pre_channel_DL_sf_real, pre_channel_DL_sf_imag], dim = -1) # (b * UE_num, pred_len, RB_num, M, 2) 
                
                out_tensor_sf = torch.complex(out_tensor[:, :, :, :, 0], out_tensor[:, :, :, :, 1]).squeeze() # (b * UE_num, pred_len, RB_num, M)
                out_tensor_sf = out_tensor_sf.unsqueeze(dim = -1) # (b * UE_num, pred_len, RB_num, M, 1)
                out_tensor_sf = torch.matmul(space_vec, out_tensor_sf).squeeze(dim = -1) # (b * UE_num, pred_len, RB_num, M)
                out_tensor_sf = out_tensor_sf.permute(0, 1, 3, 2) # (b * UE_num, pred_len, M, RB_num)
                out_tensor_sf = out_tensor_sf.unsqueeze(dim = -2) # (b * UE_num, pred_len, M, 1, RB_num)
                out_tensor_sf = torch.matmul(out_tensor_sf, delay_vec.conj()).squeeze(dim = -2) # (b * UE_num, pred_len, M, RB_num)
                out_tensor_sf = out_tensor_sf.permute(0, 1, 3, 2) # (b * UE_num, pred_len, RB_num, M)
                out_tensor_sf_real = out_tensor_sf.real
                out_tensor_sf_imag = out_tensor_sf.imag
                out_tensor_sf = torch.stack([out_tensor_sf_real, out_tensor_sf_imag], dim = -1) # (b * UE_num, pred_len, RB_num, M, 2) 
            
            sgcs = channel_SGCS(h_gt = pre_channel_DL_sf, h_recover = out_tensor_sf, b = b, UE_num = UE_num, pred_len = pred_len, RB_num = RB_num, M = M) if args.evaluate_sgcs == True else torch.zeros([pred_len,])
            se = MultiUE_SE(h_gt = pre_channel_DL_sf, h_recover = out_tensor_sf, b = b, UE_num = UE_num, pred_len = pred_len, RB_num = RB_num, M = M) if args.evaluate_se == True else torch.zeros([pred_len,])

            running_NMSE[:, 0] += nmse_loss.data.cpu()
            running_pre_norm[:, 0] += pre_norm.data.cpu()
            running_sgcs[:, 0] += sgcs.data.cpu()
            running_se[:, 0] += se.data.cpu()
        
    # average loss
    NMSE_losses = running_NMSE / batch_num
    pre_norm = running_pre_norm / batch_num
    sgcs = running_sgcs / batch_num
    se = running_se / batch_num 
    # print results
    print('Test NMSE for each time point:')
    print(NMSE_losses.T)
    print('Test prediction norm:')
    print(pre_norm.T)
    print('Test SGCS for each time point:')
    print(sgcs.T)
    print('Test spectral efficiency for each time point:')
    print(se.T)
    return NMSE_losses, sgcs, se, pre_nodes, gt_nodes


# main function
# run this file to train and evaluate the model
# OUTPUT: MATLAB data file
def main(training_time = args.training_time, epoch_num = args.epoch_num, b = args.batch_size, lr = args.lr, minlr = 1e-12):
    
    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # initialize parameters
    version_name = args.model_name + '_' + args.channel_type + '_v' + args.v + '_interval' + args.interval + args.training_data + '_snr' + args.snr
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    # print basic information
    print(version_name)
    print('device:%s' % device)
    print('batch_size:%d' % b)
    print('lr and minlr:(%e,%e)'%(lr,minlr))
    for arg in vars(args):
        logger.info(f'{arg} = {getattr(args, arg)}')

    # system parameters
    seq_len = args.seq_len # length of historical channel sequence
    pred_len = args.pred_len # length of predicted channel sequence
    M = args.M # antenna number
    UE_num = args.UE_num # UE number
    RB_num = args.RB_num # RB number
    RB_within_group = args.RB_within_group
    M_within_group = args.M_within_group
    group_num = (RB_num * M) // (RB_within_group * M_within_group) 
    
    # FFT vector
    if args.M_v == 1:
      space_vec = customized_dft(args.M) # spatial-angle domain transformation matrix, (M, M)
      space_vec = space_vec.unsqueeze(dim = 0).unsqueeze(dim = 0).unsqueeze(dim = 0) # (1, 1, 1, M, M)
      space_vec = space_vec.to(torch.complex64).to(device)
    else:
      space_vec = customized_block_diag_dft(args.M_h, args.M_v) # spatial-angle domain transformation matrix, (M, M)
      space_vec = space_vec.unsqueeze(dim = 0).unsqueeze(dim = 0).unsqueeze(dim = 0) # (1, 1, 1, M, M)
      space_vec = space_vec.to(torch.complex64).to(device)
      
    delay_vec = customized_dft(RB_num) # (RB_num, RB_num)
    delay_vec = delay_vec.unsqueeze(dim = 0).unsqueeze(dim = 0).unsqueeze(dim = 0) # (1, 1, 1, RB_num, RB_num)
    delay_vec = delay_vec.to(torch.complex64).to(device)
  
    path_eval = args.path_eval  
    eval_loader = Dataloader(path = path_eval, batch_size = b, device = device, read_uplink = args.read_uplink,
                             UE_num = UE_num, M = M, RB_num = RB_num, sum_len = seq_len + pred_len, seq_len = seq_len, pred_len = pred_len, t0 = args.t0,
                             num_workers = args.num_workers, shuffle = False)       
      
    nmse_eval = np.zeros((pred_len))
    se_eval = np.zeros((pred_len))
    sgcs_eval = np.zeros((pred_len))
    
    criterion = torch.nn.MSELoss() # MSE loss

    # model initialization
    model = Model(configs = argparse.Namespace(**{**vars(args), 'seq_len': args.seq_len // group_num if args.port_estimation_mode == 'full_long_period' else args.seq_len}))
    model.to(device)
    
    model_name = args.saved_path + args.checkpoint_name + '.pth'
    
    model.load_state_dict(torch.load(model_name, map_location = device))
    model.to(device)
    model.pred_len = pred_len
    model.eval()
       
    # eval mode, where dropout is off
    model.eval()
    # eval_losses: average loss in evaluation set     
    start = time.time()           
    NMSE_losses, sgcs, se, pre_nodes, gt_nodes = eval(model, eval_loader, UE_num, RB_num, M, seq_len, pred_len, RB_within_group, M_within_group, group_num, b, device, criterion, space_vec, delay_vec)
    print('This epoch takes %.1f s' % (time.time() - start))
    
    nmse_eval[:] = NMSE_losses.squeeze().cpu().numpy()
    sgcs_eval[:] = sgcs.squeeze().cpu().numpy()
    se_eval[:] = se.squeeze().cpu().numpy()
                              
    # save results into mat file            
    mat_name = args.saved_path + 'eval_result_' + args.training_data + '.mat'
    sio.savemat(mat_name, {
        'nmse_eval' : nmse_eval,
        'sgcs_eval' : sgcs_eval,
        'se_eval' : se_eval,
        'pre_nodes' : pre_nodes,
        'gt_nodes' : gt_nodes
    })

if __name__ == '__main__':
    main()
