from .preproccessing import WavePreproccesing, WavePreproccesingFromHDF
from torch.utils.data import Dataset, IterableDataset
import torch
import h5py
import numpy as np
import pandas as pd
from .loader.h5 import EarthQuakeWaveSlidingWindowHDF5EventOnlyDataset, NewHDF5FullDataset
from sklearn.model_selection import train_test_split

"""
BROKEN
"""


class EarthQuakeWaveSlidingWindowNumpyEventOnlyDataset(Dataset):
    def __init__(
        self,
        length,
        x,
        y,
        meta,
        stride,
        count,
        offset_pos,
        x_margin=400,
        normalize=True,
        noise_level=0.01,
    ):
        self.data_length = length
        self.x = x
        self.y = y
        self.meta = meta
        self.L = self.x.shape[1]
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos
        self.len = 0
        self.x_margin = x_margin

        self.normalize = normalize
        self.noise_level = noise_level

        self.windows = []

        start = offset_pos
        end = offset_pos + count

        for sample_idx in range(start, end):
            P = y[sample_idx, 0]
            S = y[sample_idx, 1]
            start_interval = max(0, int(P - x_margin))
            end_interval = min(self.L, int(S + x_margin))
            interval_len = end_interval - start_interval
            if interval_len >= self.data_length:
                for w_start in range(
                    start_interval, end_interval - self.data_length + 1, stride
                ):
                    self.windows.append((sample_idx, w_start))

        print(len(self.windows))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [
                self._get_single(i)
                for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)
            ]
        else:
            return self._get_single(idx)

    def _get_single(self, idx: int):
        sample_idx, x_start = self.windows[idx]
        x_end = x_start + self.data_length
        print(sample_idx)

        # window data
        x_window = torch.from_numpy(self.x[sample_idx, x_start:x_end]).float().T

        if self.normalize:
            x_window = self.__normalize(x_window)

        if self.noise_level > 0:
            if np.random.random() > 0.5:
                noise = (
                    torch.randn_like(x_window, device=x_window.device)
                    * self.noise_level
                )
                x_window += noise

        # label
        label = torch.zeros(self.data_length, dtype=torch.float32)

        P = float(self.y[sample_idx, 0])
        S = float(self.y[sample_idx, 1])

        P_in_window = P - x_start
        S_in_window = S - x_start

        label_margin = 50
        start_in_window = P_in_window - label_margin
        end_in_window = S_in_window + label_margin

        event_start = int(max(0, start_in_window))
        event_end = int(min(self.data_length, end_in_window))

        if event_start < event_end:
            label[event_start:event_end] = 1.0

        return x_window, label.unsqueeze(0)

    def __normalize(self, wave):
        mean = wave.mean(dim=1, keepdim=True)
        std = wave.std(dim=1, keepdim=True) + 1e-6
        return (wave - mean) / std


class DataLoader:
    def __init__(self, args):
        self.source = None
        if args.hdf5 is not None:
            self.source = "hdf5"
            if args.csv is None:
                raise ValueError("csv must be used with hdf5")
            self.df = pd.read_csv(args.csv)
            df_noice =self.df[self.df.trace_category == "noise"]
            df = self.df[self.df.trace_category == "earthquake_local"]
            event_ids = df["source_id"].unique()
            train_events, test_events = train_test_split(
                event_ids,
                test_size=0.2,
                shuffle=False,
            )
            self.df_train = df[df.source_id.isin(train_events)]
            self.df_test = df[df.source_id.isin(test_events)]

            if len(df_noice) > 0:
                train_noice, test_noice = train_test_split(
                    df_noice ,
                    test_size=0.2,
                    shuffle=False,
                )
                self.df_train = pd.concat(
                    [self.df_train, train_noice],
                    axis=0,
                    ignore_index=True
                )

                self.df_test = pd.concat(
                    [self.df_test, test_noice],
                    axis=0,
                    ignore_index=True
                )
                self.df_train = self.df_train.sample(
                    frac=1.0,
                    random_state=42
                ).reset_index(drop=True)

                self.df_test = self.df_test.sample(
                    frac=1.0,
                    random_state=42
                ).reset_index(drop=True)

            self.hdf5 = args.hdf5
            print("using hdf5 and csv")

        elif args.train_npz is not None:
            self.source = "npz"
            train_data = np.load(args.train_npz, mmap_mode="r", allow_pickle=True)
            test_data = np.load(args.test_npz, mmap_mode="r", allow_pickle=True)
            self.X_train = train_data["x"]
            self.y_train = train_data["y"]
            self.meta_train = train_data["meta"]
            self.X_test = test_data["x"]
            self.y_test = test_data["y"]
            self.meta_test = test_data["meta"]

        else:
            self.source = "np"
            self.X_train = np.load(args.x_train, mmap_mode="r").astype(np.float32)[:]
            self.X_test = np.load(args.x_test, mmap_mode="r").astype(np.float32)[:]
            self.y_train = np.load(args.y_train, mmap_mode="r")[:]
            self.y_test = np.load(args.y_test, mmap_mode="r")[:]
            self.meta_train = np.load(args.meta_train, allow_pickle=True)
            self.meta_test = np.load(args.meta_test, allow_pickle=True)
            print(f"X_train: {self.X_train.shape}")
            print(f"X_test: {self.X_test.shape}")

    def getDataset(self, length, stride, count, offset_pos, is_test):
        if self.source == "np" or self.source == "npz":
            if is_test:
                if count is None:
                    count = self.X_test.shape[0]
                return EarthQuakeWaveSlidingWindowNumpyEventOnlyDataset(
                    length,
                    self.X_test,
                    self.y_test,
                    self.meta_test,
                    stride,
                    count,
                    offset_pos,
                )
            else:
                if count is None:
                    count = self.X_train.shape[0]
                return EarthQuakeWaveSlidingWindowNumpyEventOnlyDataset(
                    length,
                    self.X_train,
                    self.y_train,
                    self.meta_train,
                    stride,
                    count,
                    offset_pos,
                )

        # TODO: add chunk system saat pakai hdf5
        elif self.source == "hdf5":
            df_ = self.df_test if is_test else self.df_train
            print("using HDF5 ", df_.shape)
            if count is None:
                count = len(df_)
            return NewHDF5FullDataset(
                df=df_,
                hdf5_path=self.hdf5,
            )
            # return EarthQuakeWaveSlidingWindowHDF5EventOnlyDataset(
            #     length=length,
            #     df=df_,
            #     hdf5_path=self.hdf5,
            #     count=count,
            #     stride=stride,
            #     offset_pos=offset_pos,
            # )
