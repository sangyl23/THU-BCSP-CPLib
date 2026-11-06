# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def channel_NMSE(h_gt: t.Tensor, h_pre: t.Tensor) -> t.Tensor:
    """Calculate channel NMSE for a sequence of complex matrices.

    Both inputs must have shape ``(batch, pred_len, RB_num, M)`` and a
    complex dtype.  The last two dimensions are treated as one matrix, so
    each batch/time pair contributes one matrix-Frobenius-norm NMSE.  The
    result is averaged over the batch dimension and has shape
    ``(pred_len,)``.
    """
    if h_gt.ndim != 4 or h_pre.ndim != 4:
        raise ValueError("channel_NMSE expects tensors with shape (batch, pred_len, RB_num, M)")
    if not t.is_complex(h_gt) or not t.is_complex(h_pre):
        raise TypeError("channel_NMSE expects complex-valued tensors")
    if h_gt.shape != h_pre.shape:
        raise ValueError("channel_NMSE expects inputs with the same shape")

    error_norm_sq = t.linalg.matrix_norm(h_gt - h_pre, ord="fro", dim=(-2, -1)).square()
    target_norm_sq = t.linalg.matrix_norm(h_gt, ord="fro", dim=(-2, -1)).square()
    nonzero_target = target_norm_sq > 0
    safe_target_norm_sq = t.where(nonzero_target, target_norm_sq, t.ones_like(target_norm_sq))
    nmse = (error_norm_sq / safe_target_norm_sq) * nonzero_target
    return nmse.mean(dim=0)


def channel_SGCS(h_gt: t.Tensor, h_pre: t.Tensor) -> t.Tensor:
    """Average squared generalized cosine similarity per prediction step."""
    if h_gt.shape != h_pre.shape or h_gt.ndim != 4:
        raise ValueError('channel_SGCS expects equal (batch, pred_len, RB_num, M) tensors.')
    numerator = t.abs(t.sum(h_gt.conj() * h_pre, dim=-1)).square()
    denominator = (t.linalg.vector_norm(h_gt, dim=-1).square()
                   * t.linalg.vector_norm(h_pre, dim=-1).square()).clamp_min(1e-12)
    return (numerator / denominator).mean(dim=(0, 2))


def multi_ue_se(h_gt: t.Tensor, h_pre: t.Tensor, ue_num: int,
                snr_db: float = 20.0) -> t.Tensor:
    """ZF spectral efficiency with UE folded into the leading batch dimension."""
    if h_gt.shape != h_pre.shape or h_gt.ndim != 4:
        raise ValueError('multi_ue_se expects equal (batch*UE, pred_len, RB_num, M) tensors.')
    if h_gt.shape[0] % ue_num:
        raise ValueError('Leading dimension must be divisible by ue_num.')
    batch = h_gt.shape[0] // ue_num
    pred_len, rb_num, antenna_num = h_gt.shape[1:]
    actual = h_gt.reshape(batch, ue_num, pred_len, rb_num, antenna_num).permute(0, 2, 3, 1, 4)
    predicted = h_pre.reshape(batch, ue_num, pred_len, rb_num, antenna_num).permute(0, 2, 3, 1, 4)
    actual = actual / t.linalg.vector_norm(actual, dim=-1, keepdim=True).clamp_min(1e-12)
    precoder = t.linalg.pinv(predicted)
    precoder = precoder / t.linalg.vector_norm(precoder, dim=-2, keepdim=True).clamp_min(1e-12)
    effective = t.abs(actual @ precoder).square()
    signal = t.diagonal(effective, dim1=-2, dim2=-1)
    interference = effective.sum(dim=-1) - signal
    noise = 10.0 ** (-snr_db / 10.0)
    rates = t.log2(1.0 + signal / (interference + noise))
    return rates.sum(dim=-1).mean(dim=(0, 2))


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
