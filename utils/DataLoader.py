from .preproccessing import WavePreproccesing, WavePreproccesingFromHDF
from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
import pandas as pd


class EarthQuakeWaveSlidingWindowHDF5Dataset(Dataset):
    def __init__(self, length, df, hdf5_path, stride, count, offset_pos, L=6000):
        super().__init__()
        self.data_length = length
        self.df = df
        self.hdf5_path = hdf5_path
        self.hdf5 = None
        self.L = L
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos
        
    def _ensure_hdf5_open(self):
        if self.hdf5 is None:
            # setiap worker buka file sendiri
            self.hdf5 = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return int(self.count * (((self.L - self.data_length) // self.stride) + 1))
    
    def __getitem__(self, index):
        self._ensure_hdf5_open()
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

        df = self.df.iloc[x_index]
        hdf = self.hdf5.get("data/" + str(df.trace_name))
        attrs = hdf.attrs
        xx = WavePreproccesingFromHDF(hdf).get()
        x = torch.from_numpy(xx[x_start:x_end])

        label = [0.0 for i in range(0, self.data_length)]
        p_arrived_sample = float(attrs['p_arrival_sample']) if attrs['p_arrival_sample'] not in ['', None] else 0.0
        s_arrived_sample = float(attrs['s_arrival_sample']) if attrs['s_arrival_sample'] not in ['', None] else 0.0

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

        return x.permute(1, 0), label


class EarthQuakeWaveSlidingWindowNumpyDataset(Dataset):
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
        x_ = WavePreproccesing(self.x[x_index], self.y[x_index]).get()

        x = torch.from_numpy(x_[x_start:x_end]).permute(1, 0)
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


class DataLoader:
    def __init__(self, hdf5, csv, x_test, x_train, y_test, y_train):
        self.source = None
        if hdf5 is not None:
            self.source = "hdf5"
            if csv is None:
                raise ValueError("csv must be used with hdf5")
            self.df = pd.read_csv(csv)
            self.hdf5 = h5py.File(hdf5, "r")

        elif x_test != None and x_train != None and y_test != None and y_train != None:
            self.source = "np"
            self.X_train = np.load(x_train, mmap_mode="r").astype(np.float32)[:]
            self.X_test = np.load(x_test, mmap_mode="r").astype(np.float32)[:]
            self.y_train = np.load(y_train, mmap_mode="r")[:]
            self.y_test = np.load(y_test, mmap_mode="r")[:]

        if self.source is None:
            raise ValueError("one of data source must be available")

    def getDataset(self, length, stride, count, offset_pos, is_test):
        if self.source == "np":
            if is_test:
                if count is None:
                    count = self.X_test.shape[0]
                return EarthQuakeWaveSlidingWindowNumpyDataset(
                    length, self.X_test, self.y_test, stride, count, offset_pos
                )
            else:
                if count is None:
                    count = self.X_train.shape[0]
                return EarthQuakeWaveSlidingWindowNumpyDataset(
                    length, self.X_train, self.y_train, stride, count, offset_pos
                )

        elif self.source == "hdf5":
            if count is None:
                count = self.X_test.shape[0]
            return EarthQuakeWaveSlidingWindowHDF5Dataset(
                length, self.df, self.hdf5, stride, count, offset_pos
            )
