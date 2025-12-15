import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data

"""
BROKEN
"""


class EarthQuakeWaveSlidingWindowHDF5EventOnlyDataset(Dataset):
    def __init__(
        self,
        length,
        df,
        hdf5_path,
        count,
        stride=1,
        offset_pos=0,
        x_margin=0,
        normalize=False,
        noise_level=0.4,
        windows=None,
        use_balancing: bool = False,
    ):
        self.data_length = length
        self.df = df
        self.hdf5_path = hdf5_path  # hanya simpan path, jangan buka file!
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos
        self.x_margin = x_margin
        self.normalize = normalize
        self.noise_level = noise_level
        self.use_balancing = use_balancing

        self.balancing_rules = {
            "min": 0.8,  # kelas paling kecil
            "2nd_min": 0.8,  # kelas kedua paling kecil
            "middle": 0.8,  # semua kelas di tengah
            "max": 0.8,  # kelas terbesar (undersample)
        }

        self.h5 = None  # handle kosong

        if windows is None:
            self.windows = self._build_windows()
        else:
            self.windows = windows

        # do balancing
        if use_balancing:
            self._balancing()

    def _lazy_init(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r", swmr=True, libver="latest")

    def _build_windows(self):
        start = self.offset_pos
        end = self.offset_pos + self.count
        print(f"Building windows from {start} to {end} ...")

        with h5py.File(self.hdf5_path, "r", swmr=True, libver="latest") as h5:
            L = h5["data/" + self.df.iloc[0].trace_name].shape[0]

        P = self.df.p_arrival_sample.to_numpy()
        S = self.df.s_arrival_sample.to_numpy()

        P_left = np.maximum(0, P - self.x_margin)
        P_right = np.minimum(6000, P + self.x_margin)

        S_left = np.maximum(0, S - self.x_margin)
        S_right = np.minimum(6000, S + self.x_margin)

        window_starts = np.arange(0, 6000 - self.data_length + 1, self.stride)
        window_ends = window_starts + self.data_length

        # ---------------------------
        # 2. Window valid jika masuk margin P atau S
        # ---------------------------
        P_valid = (window_starts[None, :] < P_right[:, None]) & (
            window_ends[None, :] > P_left[:, None]
        )

        S_valid = (window_starts[None, :] < S_right[:, None]) & (
            window_ends[None, :] > S_left[:, None]
        )

        # ---------------------------
        # 3. Window yang benar-benar mengandung arrival (lebih ketat)
        # ---------------------------
        P_in = (P[:, None] >= window_starts[None, :]) & (
            P[:, None] < window_ends[None, :]
        )
        S_in = (S[:, None] >= window_starts[None, :]) & (
            S[:, None] < window_ends[None, :]
        )

        P_eff = P_valid & P_in
        S_eff = S_valid & S_in

        category = {
            "P_only": 0,
            "P_and_S": 0,
            "no_P_no_S": 0,
            "S_only": 0,
        }

        category["P_and_S"] = np.sum(P_eff & S_eff)
        category["P_only"] = np.sum(P_eff & ~S_eff)
        category["S_only"] = np.sum(~P_eff & S_eff)
        category["no_P_no_S"] = np.sum((P_valid | S_valid) & ~P_eff & ~S_eff)
        self.category = category

        print(category)

        idx_p_and_s = np.where(P_eff & S_eff)
        idx_p_only = np.where(P_eff & ~S_eff)
        idx_s_only = np.where(~P_eff & S_eff)
        idx_no_p_no_s = np.where((P_valid | S_valid) & ~P_eff & ~S_eff)

        self.windows_p_and_s = list(zip(idx_p_and_s[0], idx_p_and_s[1]))
        self.windows_p_only = list(zip(idx_p_only[0], idx_p_only[1]))
        self.windows_s_only = list(zip(idx_s_only[0], idx_s_only[1]))
        self.windows_no_p_no_s = list(zip(idx_no_p_no_s[0], idx_no_p_no_s[1]))

        self.counted_dict = {
            "P_and_S": len(self.windows_p_and_s),
            "P_only": len(self.windows_p_only),
            "S_only": len(self.windows_s_only),
            "no_P_no_S": len(self.windows_no_p_no_s),
        }

        windows = np.array(
            self.windows_p_and_s
            + self.windows_p_only
            + self.windows_s_only
            + self.windows_no_p_no_s
        )

        self.labels = np.array(
            [0] * len(self.windows_p_and_s)
            + [1] * len(self.windows_p_only)
            + [2] * len(self.windows_s_only)
            + [3] * len(self.windows_no_p_no_s)
        )

        # 4. Shuffle agar tidak berurutan per kategori
        np.random.shuffle(windows)

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
        print(sample_idx)

        data = self.h5["data/" + curr_df.trace_name]
        x_end = x_start + self.data_length

        x_window = torch.from_numpy(data[x_start:x_end]).float().T

        if self.normalize:
            x_window = self._normalize(x_window)

        if self.noise_level > 0:
            self.__augmentation__(x_window)

        label = torch.zeros(self.data_length, dtype=torch.float32)

        P_in_window = curr_df.p_arrival_sample - x_start
        S_in_window = curr_df.s_arrival_sample - x_start

        margin = 50
        a = max(0, P_in_window - margin)
        b = min(self.data_length, S_in_window + margin)

        if a < b:
            label[int(a) : int(b)] = 1.0

        return x_window, label.unsqueeze(0)

    def _normalize(self, wave):
        rms = torch.sqrt(torch.mean(wave**2, dim=1, keepdim=True) + 1e-4)
        wave = wave / rms
        return wave

    def __augmentation__(self, x_window):
        if np.random.random() > 0.5:
            noise = (
                torch.randn_like(x_window, device=x_window.device) * self.noise_level
            )
            x_window += noise
        return x_window

    def __balance_samples(self, samples, target, class_name):
        n_samples = len(samples)
        if n_samples > target:
            # UNDERSAMPLE: ambil random subset
            sampled = list(np.random.choice(len(samples), target, replace=False))
            result = [samples[i] for i in sampled]
            print(f"  {class_name}: {n_samples} -> {target} (undersampled)")
        elif n_samples < target:
            # OVERSAMPLE: duplicate dengan random
            n_duplicates = target // n_samples
            n_extra = target % n_samples
            result = samples * n_duplicates
            if n_extra > 0:
                extra = list(np.random.choice(len(samples), n_extra, replace=False))
                result.extend([samples[i] for i in extra])
            print(
                f"  {class_name}: {n_samples:,} -> {target:,} (oversampled x{target / n_samples:.2f})"
            )
        else:
            result = samples
            print(f"  {class_name}: {n_samples} (unchanged)")
        return result

    def _balancing(self):
        counts = sorted(self.counted_dict.items(), key=lambda x: x[1])
        class_names = [c[0] for c in counts]
        class_counts = [c[1] for c in counts]
        total = sum(class_counts)

        max_class = class_counts[-1]  # terbesar
        min_class = class_counts[0]  # terkecil

        print("Distribusi awal:")
        for name, count in counts:
            print(f"{name}: {count} ({count / total * 100:.2f}%)")
        print("\n")

        targets = {}

        for idx, (name, count) in enumerate(counts):

            if idx == 0:
                coef = self.balancing_rules["min"]
                label = "min"

            elif idx == 1:
                coef = self.balancing_rules["2nd_min"]
                label = "2nd_min"

            elif idx == len(counts) - 1:
                coef = self.balancing_rules["max"]
                label = "max"

            else:
                coef = self.balancing_rules["middle"]
                label = "middle"

            target = int(max_class * coef)
            targets[name] = target

            print(f"Target untuk kelas {name:15} ({label:8}): {count:,} -> {target:,}")

        print("=" * 50)
        new_total = sum(targets.values())
        print("Distribusi setelah balancing:\n")

        print("\nBalancing process:")
        balanced_p_and_s = self.__balance_samples(
            self.windows_p_and_s, targets["P_and_S"], "P_and_S"
        )
        balanced_p_only = self.__balance_samples(
            self.windows_p_only, targets["P_only"], "P_only"
        )
        balanced_s_only = self.__balance_samples(
            self.windows_s_only, targets["S_only"], "S_only"
        )
        balanced_no_p_no_s = self.__balance_samples(
            self.windows_no_p_no_s, targets["no_P_no_S"], "no_P_no_S"
        )

        balanced_windows = (
            balanced_p_and_s + balanced_p_only + balanced_s_only + balanced_no_p_no_s
        )

        # 4. Shuffle agar tidak berurutan per kategori
        np.random.shuffle(balanced_windows)

        # 5. Buat label untuk setiap window
        # 0: P_and_S, 1: P_only, 2: S_only, 3: no_P_no_S
        balanced_labels = (
            [0] * len(balanced_p_and_s)
            + [1] * len(balanced_p_only)
            + [2] * len(balanced_s_only)
            + [3] * len(balanced_no_p_no_s)
        )

        # Shuffle labels sesuai dengan shuffle windows
        indices = list(range(len(balanced_windows)))
        np.random.shuffle(indices)
        balanced_windows = [balanced_windows[i] for i in indices]
        balanced_labels = [balanced_labels[i] for i in indices]

        self.windows = balanced_windows
        #
        # for name, old_count in counts:
        #     new_count = targets[name]
        #     old_ratio = old_count / total
        #     new_ratio = new_count / new_total
        #     change = "↑" if new_count > old_count else "↓"
        #     print(
        #         f"{name:15} : {old_ratio * 100:6.2f}% -> {new_ratio * 100:6.2f}% {change} ({old_count:,} -> {new_count:,})")
        #
        # print("\n")
        # print(f"Total: {total:,} -> {new_total:,}")
        # print(f"Iterasi per epoch (batch=128): {int(new_total / 128):,}")
        # print(f"Balance ratio (min/max): {min(targets.values()) / max(targets.values()) * 100:.2f}%")


"""
New Dataset
"""


class NewHDF5WindowDataset(Dataset):
    def __init__(
        self,
        length,
        df,
        hdf5_path,
        count,
        stride=500,
        offset_pos=0,
        x_margin=2000,
        normalize=False,
        noise_level=0.4,
        windows=None,
    ):
        self.data_length = length
        self.df = df
        self.hdf5_path = hdf5_path
        self.stride = stride
        self.count = count
        self.offset_pos = offset_pos
        self.x_margin = x_margin
        self.normalize = normalize
        self.noise_level = noise_level
        self.h5 = None

        if windows is None:
            self.windows, self.labels = self._build_windows()
        else:
            self.windows = windows
            self.labels = None

    def _lazy_init(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r", swmr=True, libver="latest")

    def _build_windows(self):
        start = self.offset_pos
        end = self.offset_pos + self.count
        print(f"Building windows from {start} to {end} ...")

        with h5py.File(self.hdf5_path, "r", swmr=True, libver="latest") as h5:
            L = h5["data/" + self.df.iloc[0].trace_name].shape[0]

        P = self.df.p_arrival_sample.to_numpy()
        S = self.df.s_arrival_sample.to_numpy()

        P_left = np.maximum(0, P - self.x_margin)
        P_right = np.minimum(6000, P + self.x_margin)

        S_left = np.maximum(0, S - self.x_margin)
        S_right = np.minimum(6000, S + self.x_margin)

        window_starts = np.arange(0, 6000 - self.data_length + 1, self.stride)
        window_ends = window_starts + self.data_length

        # Window valid jika masuk margin P atau S
        P_valid = (window_starts[None, :] < P_right[:, None]) & (
            window_ends[None, :] > P_left[:, None]
        )

        S_valid = (window_starts[None, :] < S_right[:, None]) & (
            window_ends[None, :] > S_left[:, None]
        )

        # Window yang benar-benar mengandung arrival
        P_in = (P[:, None] >= window_starts[None, :]) & (
            P[:, None] < window_ends[None, :]
        )
        S_in = (S[:, None] >= window_starts[None, :]) & (
            S[:, None] < window_ends[None, :]
        )

        P_eff = P_valid & P_in
        S_eff = S_valid & S_in

        # Hitung kategori
        category = {
            "P_and_S": np.sum(P_eff & S_eff),
            "P_only": np.sum(P_eff & ~S_eff),
            "S_only": np.sum(~P_eff & S_eff),
            "no_P_no_S": np.sum((P_valid | S_valid) & ~P_eff & ~S_eff),
        }

        print("Distribusi kelas:")
        for k, v in category.items():
            print(f"  {k}: {v}")

        # Ekstrak indices
        idx_p_and_s = np.where(P_eff & S_eff)
        idx_p_only = np.where(P_eff & ~S_eff)
        idx_s_only = np.where(~P_eff & S_eff)
        idx_no_p_no_s = np.where((P_valid | S_valid) & ~P_eff & ~S_eff)

        windows_p_and_s = list(zip(idx_p_and_s[0], idx_p_and_s[1]))
        windows_p_only = list(zip(idx_p_only[0], idx_p_only[1]))
        windows_s_only = list(zip(idx_s_only[0], idx_s_only[1]))
        windows_no_p_no_s = list(zip(idx_no_p_no_s[0], idx_no_p_no_s[1]))

        # Gabungkan semua windows
        all_windows = (
            windows_p_and_s + windows_p_only + windows_s_only + windows_no_p_no_s
        )

        # Buat label untuk setiap window
        # 0: P_and_S, 1: P_only, 2: S_only, 3: no_P_no_S
        all_labels = np.array(
            [0] * len(windows_p_and_s)
            + [1] * len(windows_p_only)
            + [2] * len(windows_s_only)
            + [3] * len(windows_no_p_no_s)
        )

        print(f"Total windows: {len(all_windows)}")

        return all_windows, all_labels

    def get_sample_weights(self, balancing_rules=None):

        if balancing_rules is None:
            balancing_rules = {
                "min": 0.8,
                "2nd_min": 0.8,
                "middle": 0.8,
                "max": 0.8,
            }

        # Hitung jumlah sample per kelas
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))

        # Urutkan berdasarkan jumlah
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

        # Tentukan target count untuk setiap kelas
        max_count = sorted_classes[-1][1]
        class_targets = {}

        print("\n=== Perhitungan Weight untuk Balancing ===")
        for idx, (label, count) in enumerate(sorted_classes):
            if idx == 0:
                coef = balancing_rules["min"]
                label_str = "min"
            elif idx == 1:
                coef = balancing_rules["2nd_min"]
                label_str = "2nd_min"
            elif idx == len(sorted_classes) - 1:
                coef = balancing_rules["max"]
                label_str = "max"
            else:
                coef = balancing_rules["middle"]
                label_str = "middle"

            target = max_count * coef
            class_targets[label] = target

            class_name = ["P_and_S", "P_only", "S_only", "no_P_no_S"][label]
            print(
                f"Kelas {class_name:15} ({label_str:8}): {count:,} samples, target weight: {target/count:.4f}"
            )

        # Hitung weight untuk setiap sample
        # Weight = target_count / actual_count
        # Semakin kecil kelasnya, semakin besar weightnya
        sample_weights = np.zeros(len(self.labels))
        for label in unique_labels:
            mask = self.labels == label
            weight = class_targets[label] / class_counts[label]
            sample_weights[mask] = weight

        print(
            f"\nTotal effective samples setelah weighting: {sample_weights.sum():.0f}"
        )
        print("=" * 50)

        return torch.DoubleTensor(sample_weights)

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
            x_window = self._augmentation(x_window)

        label = torch.zeros(self.data_length, dtype=torch.float32)

        P_in_window = curr_df.p_arrival_sample - x_start
        S_in_window = curr_df.s_arrival_sample - x_start

        margin = 50
        a = max(0, P_in_window - margin)
        b = min(self.data_length, S_in_window + margin)

        if a < b:
            label[int(a) : int(b)] = 1.0

        return x_window, label.unsqueeze(0)

    def _normalize(self, wave):
        rms = torch.sqrt(torch.mean(wave**2, dim=1, keepdim=True) + 1e-4)
        wave = wave / rms
        return wave

    def _augmentation(self, x_window):
        if np.random.random() > 0.5:
            noise = torch.randn_like(x_window) * self.noise_level
            x_window = x_window + noise
        return x_window


class NewHDF5FullDataset(Dataset):
    def __init__(self, df: pd.DataFrame, hdf5_path: str):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.df = df
        self.hdf5_instance: h5py.File = None
        self.noise_level = 0.4
        self.__lazy_init()

    def __lazy_init(self):
        if self.hdf5_instance is None:
            self.hdf5_instance = h5py.File(
                self.hdf5_path, "r", swmr=True, libver="latest"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_df = self.df.iloc[idx]
        wave = self.hdf5_instance["data/" + current_df.trace_name]
        x_window = torch.from_numpy(wave[:]).float().T

        if np.random.random() > 0.5:
            # 1. Gaussian Noise
            noise = (
                torch.randn_like(x_window, device=x_window.device) * self.noise_level
            )
            x_window += noise

        if np.random.random() > 0.5:
            # 2. Random amplitude scaling
            scale = 0.8 + 0.4 * torch.rand(
                1, device=x_window.device
            )  # skala antara 0.8 - 1.2
            x_window *= scale

        if np.random.random() > 0.5:
            # 3. Random time shift
            shift = int(torch.randint(-5, 6, (1,)).item())  # geser ±5 sample
            x_window = torch.roll(x_window, shifts=shift, dims=1)

        if np.random.random() > 0.5:
            # 4. Time masking (sebagian window diset 0)
            t = x_window.shape[1]
            mask_width = int(0.1 * t)  # mask 10% dari panjang
            start = int(torch.randint(0, t - mask_width, (1,)).item())
            x_window[:, start : start + mask_width] = 0

        # Normalisasi setelah augmentasi
        x_window = self._normalize(x_window)

        raw_p = wave.attrs.get("p_arrival_sample")
        P = int(raw_p) if raw_p not in (None, "", b"") else -1

        raw_s = wave.attrs.get("s_arrival_sample")
        S = int(raw_s) if raw_s not in (None, "", b"") else -1

        P = max(0, min(P, 6000))
        S = max(0, min(S, 6000))

        label = torch.zeros(6000, dtype=torch.float32)
        label[P:S] = 1.0

        return x_window, label.unsqueeze(0)

    def _normalize(self, wave):
        # Hitung mean dan std per batch (dim=1)
        mean = torch.mean(wave, dim=1, keepdim=True)
        std = (
            torch.std(wave, dim=1, keepdim=True) + 1e-8
        )  # tambahkan epsilon agar tidak divisi 0

        # Normalisasi
        wave = (wave - mean) / std

        # Clamp output untuk mencegah nilai ekstrem jika perlu
        wave = torch.clamp(wave, min=-10, max=10)

        return wave


class HDF5PhaseGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        df,
        hdf5_path,
        data_length=200,
        stride=500,
        x_margin=100,
        transform=None,
        pre_transform=None,
    ):
        self.df = df
        self.hdf5_path = hdf5_path
        self.data_length = data_length
        self.stride = stride
        self.x_margin = x_margin
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # tidak digunakan, karena kita load dari hdf5 langsung
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []

        with h5py.File(self.hdf5_path, "r", swmr=True, libver="latest") as h5:
            for idx, row in self.df.iterrows():
                # ambil waktu pick dan posisi stasiun
                t_P = row.p_arrival_sample
                t_S = row.s_arrival_sample
                lat = row.receiver_latitude
                lon = row.receiver_longitude

                # buat fitur node: [t_P, t_S, lat, lon]
                x = torch.tensor([[t_P, t_S, lat, lon]], dtype=torch.float)

                # optional: edge_index, nanti bisa dibuat dinamis
                edge_index = torch.empty((2, 0), dtype=torch.long)  # sementara kosong

                # buat label sederhana, misal P/S presence
                label = torch.tensor(
                    [1 if not np.isnan(t_P) or not np.isnan(t_S) else 0],
                    dtype=torch.float,
                )

                data = Data(x=x, edge_index=edge_index, y=label)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
