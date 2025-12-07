from .preproccessing import WavePreproccesing, WavePreproccesingFromHDF
from torch.utils.data import Dataset, IterableDataset
import torch
import h5py
import numpy as np
import pandas as pd
import math


class EarthQuakeWaveSlidingWindowHDF5IterableDataset(IterableDataset):
    def __init__(self, length, df, hdf5_path, stride, count, offset_pos, L=6000):
        super().__init__()
        self.length = length
        self.df = df
        self.hdf5_path = hdf5_path
        self.hdf5 = None
        self.L = L
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos

        # perhitungan jumlah slide per trace
        self.slide_count = (self.L - self.length) // self.stride + 1

    def _ensure_hdf5_open(self):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.hdf5_path, "r")

    def _get_single(self, index: int):
        self._ensure_hdf5_open()

        slide_count = self.slide_count
        x_index = (index // slide_count) + self.offset_pos
        x_start = (index % slide_count) * self.stride
        x_end = x_start + self.length

        df = self.df.iloc[x_index]
        hdf = self.hdf5.get("data/" + str(df.trace_name))
        attrs = hdf.attrs
        data = np.array(hdf)[:, :3]

        # xx = WavePreproccesingFromHDF(hdf).get()
        x = torch.from_numpy(data[x_start:x_end])

        label = [0.0 for _ in range(self.length)]

        p_arrived_sample = (
            float(attrs["p_arrival_sample"])
            if attrs["p_arrival_sample"] not in ["", None]
            else 0.0
        )
        s_arrived_sample = (
            float(attrs["s_arrival_sample"])
            if attrs["s_arrival_sample"] not in ["", None]
            else 0.0
        )

        event_start = p_arrived_sample
        event_end = s_arrived_sample

        start = int(event_start - x_start if event_start >= x_start else 0)
        if start:
            end = int(event_end - x_start if event_end <= x_end else x_end - x_start)
            label = torch.zeros(self.length, dtype=torch.float32)
            if start < end:
                label[start:end] = 1.0

        label = torch.tensor([label], dtype=torch.float32)

        return x.permute(1, 0), label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single process
            start = 0
            end = self.count
        else:
            # bagi data per worker agar tidak overlap
            per_worker = int(math.ceil(self.count / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.count)

        # generate indeks sliding streaming
        for trace_i in range(start, end):
            base_index = trace_i * self.slide_count
            for s in range(self.slide_count):
                yield self._get_single(base_index + s)


class EarthQuakeWaveSlidingWindowNumpyDataset(Dataset):
    """
    @deprecated
    """

    def __init__(self, length, x, y, stride, count, offset_pos):
        self.data_length = length
        self.x = x
        self.y = y
        self.L = self.x.shape[1]
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos

    def __len__(self):
        return int(self.count * (((self.L - self.data_length) // self.stride) + 1))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start_i = index.start or 0
            stop_i = index.stop or len(self)
            step_i = index.step or 1

            items = []
            for i in range(start_i, stop_i, step_i):
                items.append(self._get_single(i))
            return items

        else:
            return self._get_single(index)

    def _get_single(self, index: int):
        slide_count = (self.L - self.data_length) // self.stride + 1
        x_index = (index // slide_count) + self.offset_pos
        x_start = (index % slide_count) * self.stride
        x_end = x_start + self.data_length
        # print(x_index, slide_count, x_start, x_end)

        # preproccessing
        # x_ = WavePreproccesing(self.x[x_index], self.y[x_index]).get()

        x = torch.from_numpy(self.x[x_index, x_start:x_end]).permute(1, 0)
        y__ = self.y[x_index]

        label = [0.0 for i in range(0, self.data_length)]
        p_arrived_sample = float(y__[0])
        s_arrived_sample = float(y__[1])
        # print(y__)
        event_start = p_arrived_sample
        event_end = s_arrived_sample

        start = int(event_start - x_start if event_start >= x_start else 0)
        if start:
            end = int(event_end - x_start if event_end <= x_end else x_end - x_start)
            # print(x_start, x_end, start, end)
            # print(len(label))

            for j in range(start, end):
                label[j] = 1.0

        found_earthquake = (x_start < event_end) and (x_end > event_start)
        found_earthquake = found_earthquake or (x_start < event_start < x_end)

        label = torch.tensor([label], dtype=torch.float32)
        return x, label


class EarthQuakeWaveSlidingWindowNumpyEventOnlyDataset(Dataset):
    def __init__(self, length, x, y, meta, stride, count, offset_pos, x_margin=500):
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

        self.windows = []

        self.data_x = x[offset_pos : offset_pos + count ]
        self.data_y = y[offset_pos : offset_pos + count]
        for sample_idx in range(len(self.data_x)):
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

        # window data
        x_window = torch.from_numpy(
            self.x[sample_idx, x_start:x_end]
        ).float().T

        # label
        label = torch.zeros(self.data_length, dtype=torch.float32)

        P = float(self.y[sample_idx, 0])
        S = float(self.y[sample_idx, 1])

        # event hanya di sekitar P-S, margin untuk label kecil saja
        label_margin = 50     # misal 50 sample, bebas kamu atur

        event_start = max(int(P - label_margin), 0)
        event_end   = min(int(S + label_margin), self.x.shape[1])

        # hitung overlap window dengan event
        start_idx = max(0, event_start - x_start)
        end_idx   = min(self.data_length, event_end - x_start)

        if start_idx < end_idx:
            label[start_idx:end_idx] = 1.0

        return x_window, label.unsqueeze(0)


class DataLoader:
    def __init__(self, args):
        self.source = None
        if args.hdf5 is not None:
            self.source = "hdf5"
            if args.csv is None:
                raise ValueError("csv must be used with hdf5")
            self.df = pd.read_csv(args.csv)
            self.hdf5 = args.hdf5
            print("using hdf5 and csv")
            
        elif args.train_npz is not None:
            self.source = "npz"
            train_data =  np.load(args.train_npz, mmap_mode="r", allow_pickle=True)
            test_data =  np.load(args.test_npz, mmap_mode="r", allow_pickle=True)
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
            if count is None:
                count = self.X_test.shape[0]
            return EarthQuakeWaveSlidingWindowHDF5IterableDataset(
                length, self.df, self.hdf5, stride, count, offset_pos
            )
