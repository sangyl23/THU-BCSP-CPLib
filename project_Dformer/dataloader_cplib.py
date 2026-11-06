from os import listdir
from os.path import isfile, join
from collections import OrderedDict
import math
import random
import re

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader as TorchLoader
from torch.utils.data import Dataset


class MatDataset(Dataset):
    def __init__(self, path='', read_uplink=False,
                 UE_num=4, M=32, RB_num=8, sum_len=36, seq_len=32,
                 pred_len=4, t0=32, return_conditions=False,
                 max_files=None, cache_size=2):
        """Load downlink/uplink CSI and optional condition variables from MAT files."""
        if max_files is not None:
            if max_files < 1:
                raise ValueError('max_files must be positive when provided.')
            dataset_indices = [
                int(match.group(1))
                for filename in listdir(path)
                if isfile(join(path, filename))
                for match in [re.fullmatch(r'dataset_(\d+)\.mat', filename)]
                if match is not None
            ]
            if not dataset_indices:
                raise FileNotFoundError(f'No dataset_x.mat files found in {path!r}.')
            first_index = min(dataset_indices)
            files = [
                join(path, f'dataset_{index}.mat')
                for index in range(first_index, first_index + max_files)
            ]
            missing_files = [filename for filename in files if not isfile(filename)]
            if missing_files:
                last_index = first_index + max_files - 1
                raise FileNotFoundError(
                    f'Expected dataset_{first_index}.mat through dataset_{last_index}.mat '
                    f'in {path!r}; missing {missing_files[0]!r}.'
                )
        else:
            files = [
                join(path, filename)
                for filename in listdir(path)
                if isfile(join(path, filename)) and filename.endswith('.mat')
            ]
            files.sort(key=self._file_sort_key)
        self.files = files
        if not self.files:
            raise FileNotFoundError(f'No .mat files found in {path!r}.')

        self.read_uplink = read_uplink
        self.UE_num, self.M, self.RB_num = UE_num, M, RB_num
        self.sum_len = sum_len
        self.seq_len, self.pred_len, self.t0 = seq_len, pred_len, t0
        self.return_conditions = return_conditions
        if cache_size < 1:
            raise ValueError('cache_size must be positive.')
        self.cache_size = cache_size
        self._file_cache = OrderedDict()

        if seq_len < 1 or pred_len < 1:
            raise ValueError('seq_len and pred_len must be positive.')
        if t0 < seq_len or t0 + pred_len > sum_len:
            raise ValueError(
                f't0={t0} with seq_len={seq_len} and pred_len={pred_len} '
                f'does not fit within sum_len={sum_len}.'
            )

        self.index_map = []
        self.indices_by_file = []
        self.file_conditions = []
        condition_names = [
            'speed_label', 'TimeInterval',
            'speed_label_norm', 'TimeInterval_norm',
        ]
        required_channels = ['H_DL_noise', 'H_DL_gt']
        if read_uplink:
            required_channels.extend(['H_UL_noise', 'H_UL_gt'])

        for file_index, filename in enumerate(self.files):
            variable_shapes = {
                name: shape for name, shape, _ in sio.whosmat(filename)
            }
            missing_channels = [
                name for name in required_channels if name not in variable_shapes
            ]
            if missing_channels:
                raise KeyError(f'{filename!r} must contain {missing_channels[0]}.')

            expected_shape = (UE_num, sum_len, RB_num, M)
            expected_shape_m1 = (UE_num, sum_len, RB_num)
            reference_shape = variable_shapes['H_DL_noise']
            for channel_name in required_channels:
                channel_shape = variable_shapes[channel_name]
                if channel_shape != reference_shape:
                    raise ValueError(f'{filename!r} has different channel-array shapes.')
                if channel_shape[1:] != expected_shape and not (
                    M == 1 and channel_shape[1:] == expected_shape_m1
                ):
                    raise ValueError(
                        f'{filename!r} has shape {channel_shape}; expected '
                        f'(batch, {UE_num}, {sum_len}, {RB_num}, {M}).'
                    )

            data = sio.loadmat(filename, variable_names=condition_names)
            conditions = {}
            for condition_name in condition_names:
                condition = data.get(condition_name)
                if condition is None:
                    if self.return_conditions:
                        raise KeyError(f'{filename!r} must contain {condition_name}.')
                    conditions[condition_name] = None
                    continue
                condition = np.asarray(condition).reshape(-1)
                if condition.shape[0] != reference_shape[0]:
                    raise ValueError(
                        f'{filename!r} has an invalid {condition_name} length.'
                    )
                conditions[condition_name] = condition

            self.file_conditions.append(conditions)
            file_indices = []
            for sample_index in range(reference_shape[0]):
                file_indices.append(len(self.index_map))
                self.index_map.append((file_index, sample_index))
            self.indices_by_file.append(file_indices)

    @staticmethod
    def _file_sort_key(filename):
        match = re.search(r'dataset_(\d+)\.mat$', filename)
        return (0, int(match.group(1))) if match else (1, filename)

    def __len__(self):
        return len(self.index_map)

    def _load_channels(self, file_index):
        """Load a MAT file once per dataset process and reuse it from an LRU cache."""
        if file_index in self._file_cache:
            channels = self._file_cache.pop(file_index)
            self._file_cache[file_index] = channels
            return channels

        variable_names = ['H_DL_noise', 'H_DL_gt']
        if self.read_uplink:
            variable_names.extend(['H_UL_noise', 'H_UL_gt'])
        data = sio.loadmat(self.files[file_index], variable_names=variable_names)
        channels = tuple(data[name] for name in variable_names)
        self._file_cache[file_index] = channels
        if len(self._file_cache) > self.cache_size:
            self._file_cache.popitem(last=False)
        return channels

    def __getitem__(self, index):
        file_index, sample_index = self.index_map[index]
        channels = self._load_channels(file_index)
        H_DL_noise, H_DL_gt = channels[:2]
        H_DL_noise = H_DL_noise[sample_index]
        H_DL_gt = H_DL_gt[sample_index]

        if self.M == 1 and H_DL_noise.ndim == 3:
            H_DL_noise = np.expand_dims(H_DL_noise, axis=-1)
            H_DL_gt = np.expand_dims(H_DL_gt, axis=-1)

        history_DL = H_DL_noise[:, self.t0 - self.seq_len:self.t0]
        target_DL = H_DL_gt[:, self.t0:self.t0 + self.pred_len]
        history_DL = np.stack((history_DL.real, history_DL.imag), axis=-1)
        target_DL = np.stack((target_DL.real, target_DL.imag), axis=-1)
        result = (
            torch.from_numpy(history_DL).float(),
            torch.from_numpy(target_DL).float(),
        )

        if self.read_uplink:
            H_UL_noise, H_UL_gt = channels[2:]
            H_UL_noise = H_UL_noise[sample_index]
            H_UL_gt = H_UL_gt[sample_index]
            if self.M == 1 and H_UL_noise.ndim == 3:
                H_UL_noise = np.expand_dims(H_UL_noise, axis=-1)
                H_UL_gt = np.expand_dims(H_UL_gt, axis=-1)
            history_UL = H_UL_noise[:, self.t0 - self.seq_len:self.t0]
            target_UL = H_UL_gt[:, self.t0:self.t0 + self.pred_len]
            history_UL = np.stack((history_UL.real, history_UL.imag), axis=-1)
            target_UL = np.stack((target_UL.real, target_UL.imag), axis=-1)
            result = (
                result[0], torch.from_numpy(history_UL).float(),
                result[1], torch.from_numpy(target_UL).float(),
            )

        if self.return_conditions:
            conditions = self.file_conditions[file_index]
            result += tuple(
                torch.tensor(np.float32(conditions[name][sample_index]))
                for name in (
                    'speed_label', 'TimeInterval',
                    'speed_label_norm', 'TimeInterval_norm',
                )
            )
        return result


class FileBatchSampler:
    """Form batches from one MAT file at a time to make file caching effective."""

    def __init__(self, dataset, batch_size, shuffle):
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        self.indices_by_file = dataset.indices_by_file
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        file_order = list(range(len(self.indices_by_file)))
        if self.shuffle:
            random.shuffle(file_order)
        for file_index in file_order:
            indices = self.indices_by_file[file_index].copy()
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]

    def __len__(self):
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.indices_by_file
        )


class Dataloader:
    def __init__(self, path='', batch_size=16, device='cpu', read_uplink=False,
                 UE_num=4, M=32, RB_num=8, sum_len=36, seq_len=32,
                 pred_len=4, t0=32, return_conditions=False,
                 num_workers=0, shuffle=False, max_files=None, cache_size=2,
                 persistent_workers=True, prefetch_factor=1):
        """Wrap MatDataset with a PyTorch loader and move tensors to device."""
        self.device = device
        self.read_uplink = read_uplink
        self.return_conditions = return_conditions
        dataset = MatDataset(
            path=path, read_uplink=read_uplink, UE_num=UE_num, M=M,
            RB_num=RB_num, sum_len=sum_len, seq_len=seq_len,
            pred_len=pred_len, t0=t0, return_conditions=return_conditions,
            max_files=max_files, cache_size=cache_size,
        )
        loader_kwargs = {
            'dataset': dataset,
            'batch_sampler': FileBatchSampler(dataset, batch_size, shuffle),
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = persistent_workers
            loader_kwargs['prefetch_factor'] = prefetch_factor
        self.loader = TorchLoader(**loader_kwargs)
        self.iterator = None

    def next_batch(self):
        try:
            if self.iterator is None:
                self.iterator = iter(self.loader)
            batch = next(self.iterator)
            batch = tuple(item.to(self.device, non_blocking=True) for item in batch)
            return batch + (False,)
        except StopIteration:
            self.iterator = None
            channel_count = 4 if self.read_uplink else 2
            condition_count = 4 if self.return_conditions else 0
            return (None,) * (channel_count + condition_count) + (True,)
