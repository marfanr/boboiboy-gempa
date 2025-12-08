import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class EarthQuakeWaveSlidingWindowHDF5EventOnlyDataset(Dataset):
    def __init__(
        self,
        length,
        df,
        hdf5_path,
        stride,
        count,
        offset_pos=0,
        x_margin=50,
        normalize=True,
        noise_level=0.01,
        windows=None,
    ):
        self.data_length = length
        self.df = df
        self.hdf5_path = hdf5_path       # hanya simpan path, jangan buka file!
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos
        self.x_margin = x_margin
        self.normalize = normalize
        self.noise_level = noise_level

        self.h5 = None                   # handle kosong

        if windows is None:
            # bangun window TANPA menyentuh HDF5
            self.windows = self._build_windows()
        else:
            self.windows = windows

    def _lazy_init(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r", swmr=True, libver="latest")

    def _build_windows(self):
        start = self.offset_pos
        end = self.offset_pos + self.count
        print(f"Building windows from {start} to {end} ...")

        # BACA SATU SAMPLE DENGAN HDF5 SAFE
        with h5py.File(self.hdf5_path, "r", swmr=True, libver="latest") as h5:
            L = h5["data/" + self.df.iloc[0].trace_name].shape[0]

        P = self.df.p_arrival_sample.to_numpy()
        S = self.df.s_arrival_sample.to_numpy()

        start_intervals = np.maximum(0, P - self.x_margin).astype(int)
        end_intervals = np.minimum(L, S + self.x_margin).astype(int)
        lengths = end_intervals - start_intervals

        valid_idx = np.where(lengths >= self.data_length)[0]

        windows = []
        for idx in valid_idx:
            st = start_intervals[idx]
            ed = end_intervals[idx]
            for w_start in range(st, ed - self.data_length + 1, self.stride):
                windows.append((idx, w_start))

        print(f"window len: {len(windows)}")
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        self._lazy_init()
        return self._get_single(idx)

    def _get_single(self, idx):
        sample_idx, x_start = self.windows[idx]
        curr_df = self.df.iloc[sample_idx]

        data = self.h5["data/" + curr_df.trace_name]
        x_end = x_start + self.data_length

        x_window = torch.from_numpy(data[x_start:x_end]).float().T

        if self.normalize:
            x_window = self._normalize(x_window)

        if self.noise_level > 0:
            if np.random.random() > 0.5:
                x_window += torch.randn_like(x_window) * self.noise_level

        label = torch.zeros(self.data_length, dtype=torch.float32)

        P_in_window = curr_df.p_arrival_sample - x_start
        S_in_window = curr_df.s_arrival_sample - x_start

        margin = 50
        a = max(0, P_in_window - margin)
        b = min(self.data_length, S_in_window + margin)

        if a < b:
            label[int(a):int(b)] = 1.0

        return x_window, label.unsqueeze(0)

    def _normalize(self, wave):
        mean = wave.mean(dim=1, keepdim=True)
        std = wave.std(dim=1, keepdim=True) + 1e-6
        return (wave - mean) / std
