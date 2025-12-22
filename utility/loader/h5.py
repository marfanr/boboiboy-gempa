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

from typing import Literal


class NewHDF5WindowDataset(Dataset):
    def __init__(
        self,
        length,
        df,
        hdf5_path,
        count,
        stride=500,
        offset_pos=0,
        x_margin=1000,
        normalize=False,
        noise_level=0.4,
        windows=None,
        balance_strategy: Literal[
            "undersample",
            "oversample",
            "weighted",
        ] = "undersample",  # "undersample", "oversample", atau "weighted"
        target_distribution=None,  # Dict untuk distribusi target, misal {"P_and_S": 0.25, "P_only": 0.25, ...}
        output_type: Literal["gempa","phase_p"] = "gempa",
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
        self.balance_strategy = balance_strategy
        self.h5 = None

        # Default target distribution: semua kelas sama banyak
        if target_distribution is None:
            self.target_distribution = {
                "P_and_S": 0.2,
                "P_only": 0.15,
                "S_only": 0.6,
                "no_P_no_S": 0.6,
            }
        else:
            self.target_distribution = target_distribution

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

            # Ekstrak indices untuk setiap kategori
            idx_p_and_s = np.where(P_eff & S_eff)
            idx_p_only = np.where(P_eff & ~S_eff)
            idx_s_only = np.where(~P_eff & S_eff)
            idx_no_p_no_s = np.where((P_valid | S_valid) & ~P_eff & ~S_eff)

            # Konversi ke list of tuples
            windows_dict = {
                "P_and_S": list(zip(idx_p_and_s[0], idx_p_and_s[1])),
                "P_only": list(zip(idx_p_only[0], idx_p_only[1])),
                "S_only": list(zip(idx_s_only[0], idx_s_only[1])),
                "no_P_no_S": list(zip(idx_no_p_no_s[0], idx_no_p_no_s[1])),
            }

            # Print distribusi awal
            print("\n=== Distribusi Awal ===")
            for k, v in windows_dict.items():
                print(f" {k}: {len(v)}")

            # Balance windows
            balanced_windows, balanced_labels = self._balance_windows(windows_dict)

            print(f"\nTotal windows setelah balancing: {len(balanced_windows)}")
            return balanced_windows, balanced_labels

    def _balance_windows(self, windows_dict):
        """Balance windows berdasarkan strategy yang dipilih"""

        class_counts = {k: len(v) for k, v in windows_dict.items()}

        if self.balance_strategy == "undersample":
            return self._undersample(windows_dict, class_counts)
        elif self.balance_strategy == "oversample":
            return self._oversample(windows_dict, class_counts)
        else:  # weighted - tidak perlu balance di sini
            return self._no_balance(windows_dict)

    def _undersample(self, windows_dict, class_counts):
        """Undersample kelas mayoritas agar seimbang dengan target distribution"""

        # Hitung target count berdasarkan kelas minoritas dan target distribution
        min_count = min(class_counts.values())

        # Hitung target count untuk setiap kelas
        target_counts = {}
        for class_name, ratio in self.target_distribution.items():
            # Target dihitung relatif terhadap min_count
            # Jika semua ratio sama (0.25), semua akan sama dengan min_count
            target_counts[class_name] = int(
                min_count / min(self.target_distribution.values()) * ratio
            )

        print("\n=== Undersampling ===")
        balanced_windows = []
        balanced_labels = []
        label_map = {"P_and_S": 0, "P_only": 1, "S_only": 2, "no_P_no_S": 3}

        for class_name, windows in windows_dict.items():
            target = target_counts[class_name]
            actual = len(windows)

            if actual > target:
                # Random sampling tanpa replacement
                sampled_indices = np.random.choice(actual, target, replace=False)
                sampled_windows = [windows[i] for i in sampled_indices]
            else:
                sampled_windows = windows

            balanced_windows.extend(sampled_windows)
            balanced_labels.extend([label_map[class_name]] * len(sampled_windows))

            print(f" {class_name:15}: {actual:,} -> {len(sampled_windows):,}")

        # Shuffle
        indices = np.random.permutation(len(balanced_windows))
        balanced_windows = [balanced_windows[i] for i in indices]
        balanced_labels = np.array(balanced_labels)[indices]

        return balanced_windows, balanced_labels

    def _oversample(self, windows_dict, class_counts):
        """Oversample kelas minoritas agar seimbang dengan target distribution"""

        # Hitung target count berdasarkan kelas mayoritas dan target distribution
        max_count = max(class_counts.values())

        # Hitung target count untuk setiap kelas
        target_counts = {}
        for class_name, ratio in self.target_distribution.items():
            target_counts[class_name] = int(
                max_count / max(self.target_distribution.values()) * ratio
            )

        print("\n=== Oversampling ===")
        balanced_windows = []
        balanced_labels = []
        label_map = {"P_and_S": 0, "P_only": 1, "S_only": 2, "no_P_no_S": 3}

        for class_name, windows in windows_dict.items():
            target = target_counts[class_name]
            actual = len(windows)

            if actual < target:
                # Random sampling dengan replacement
                sampled_indices = np.random.choice(actual, target, replace=True)
                sampled_windows = [windows[i] for i in sampled_indices]
            else:
                sampled_windows = windows

            balanced_windows.extend(sampled_windows)
            balanced_labels.extend([label_map[class_name]] * len(sampled_windows))

            print(f" {class_name:15}: {actual:,} -> {len(sampled_windows):,}")

        # Shuffle
        indices = np.random.permutation(len(balanced_windows))
        balanced_windows = [balanced_windows[i] for i in indices]
        balanced_labels = np.array(balanced_labels)[indices]

        return balanced_windows, balanced_labels

    def _no_balance(self, windows_dict):
        """Tidak melakukan balancing, gunakan weighted sampling di DataLoader"""

        print("\n=== Menggunakan Weighted Sampling (no resampling) ===")
        all_windows = []
        all_labels = []
        label_map = {"P_and_S": 0, "P_only": 1, "S_only": 2, "no_P_no_S": 3}

        for class_name, windows in windows_dict.items():
            all_windows.extend(windows)
            all_labels.extend([label_map[class_name]] * len(windows))
            print(f" {class_name:15}: {len(windows):,}")

        return all_windows, np.array(all_labels)

    def get_sample_weights(self, method="inverse"):
        """
        Hitung sample weights untuk WeightedRandomSampler

        method:
        - "inverse": weight = 1 / count (kelas minoritas dapat weight lebih besar)
        - "sqrt_inverse": weight = 1 / sqrt(count) (lebih soft)
        - "effective": effective number of samples (paper Cui et al. 2019)
        """

        if self.labels is None:
            raise ValueError(
                "Labels tidak tersedia. Pastikan _build_windows sudah dipanggil."
            )

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))

        print("\n=== Perhitungan Sample Weights ===")
        print(f"Method: {method}")

        # Hitung weight untuk setiap kelas
        class_weights = {}

        if method == "inverse":
            total_samples = len(self.labels)
            for label, count in class_counts.items():
                class_weights[label] = total_samples / (len(class_counts) * count)

        elif method == "sqrt_inverse":
            total_samples = len(self.labels)
            for label, count in class_counts.items():
                class_weights[label] = np.sqrt(
                    total_samples / (len(class_counts) * count)
                )

        elif method == "effective":
            # Effective Number of Samples: ENs = (1 - beta^n) / (1 - beta)
            beta = 0.9999
            for label, count in class_counts.items():
                effective_num = (1.0 - beta**count) / (1.0 - beta)
                class_weights[label] = 1.0 / effective_num

        # Apply weights ke setiap sample
        sample_weights = np.zeros(len(self.labels))
        for label in unique_labels:
            mask = self.labels == label
            sample_weights[mask] = class_weights[label]

            class_name = ["P_and_S", "P_only", "S_only", "no_P_no_S"][label]
            print(
                f"{class_name:15}: count={class_counts[label]:,}, weight={class_weights[label]:.4f}"
            )

        print(f"\nTotal effective samples: {sample_weights.sum():.0f}")
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


class H5MultiStationWindowDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            hdf5_path: str,
            max_length: int = 6000,
            stride: int = 500,
            length: int = 1000,
            x_margin: int = 100,
            windows: np.ndarray | None = None,
    ):
        super().__init__()

        self.hdf5_path = hdf5_path
        self.max_length = max_length
        self.stride = stride
        self.window_size = length
        self.x_margin = x_margin
        self.h5 = None

        # ============================
        # Filter event (>=3 stasiun)
        # ============================
        counts = df["source_id"].value_counts()
        self.df = df[df["source_id"].map(counts) > 2].copy()

        self.Y_MEAN = [-112.80365782, 33.71483372]
        self.Y_STD = [41.09888086, 11.07947129]

        # ============================
        # Map event
        # ============================
        source_ids = self.df["source_id"].unique()
        event_map = {sid: i for i, sid in enumerate(source_ids)}
        self.df["event_idx"] = self.df["source_id"].map(event_map)

        # Group df per event (dipakai di __getitem__)
        self.event_groups = {
            eid: g for eid, g in self.df.groupby("event_idx")
        }

        # Build windows
        self.windows = windows if windows is not None else self.build_window()

    # ------------------------------------------------
    def _lazy_init(self):
        if self.h5 is None:
            self.h5 = h5py.File(
                self.hdf5_path, "r", swmr=True, libver="latest"
            )

    # ------------------------------------------------
    def __len__(self):
        return len(self.windows)

    # ------------------------------------------------
    def __getitem__(self, idx):
        self._lazy_init()

        event_idx, win_start = self.windows[idx]
        win_end = win_start + self.window_size

        event_df = self.event_groups[event_idx]

        X_list = []
        cords_list = []
        y_list = []
        target_coords_list = []
        precursor_list = []


        for row in event_df.itertuples(index=False):
            trace = self.h5["/data/" + row.trace_name]
            wave = trace[win_start:win_end]
            wave = np.asarray(wave)

            if wave.ndim == 1:
                wave = wave[:, None]

            # ----- label per stasiun -----
            p = row.p_arrival_sample
            s = row.s_arrival_sample
            long = row.receiver_longitude
            lat = row.receiver_latitude
            elev = row.receiver_elevation_m

            target_long = row.source_longitude
            target_lat = row.source_latitude

            p_left = max(0, p - self.x_margin)
            p_right = min(self.max_length, p + self.x_margin)
            s_left = max(0, s - self.x_margin)
            s_right = min(self.max_length, s + self.x_margin)

            p_valid = (win_start < p_right) and (win_end > p_left)
            s_valid = (win_start < s_right) and (win_end > s_left)

            p_in = (p >= win_start) and (p < win_end)
            s_in = (s >= win_start) and (s < win_end)

            # print(win_start, p_in, s_in)

            # if s_valid and s_in:
            #     print(s - win_start)
            #
            # if p_valid and s_valid and p_in and s_in:
            #     label = 0  # P_and_S
            # elif p_valid and p_in:
            #     label = 1  # P_only
            # elif s_valid and s_in:
            #     label = 2  # S_only
            # elif p_valid or s_valid:
            #     label = 3  # no_P_no_S
            # else:
            #     continue  # stasiun ini tidak relevan

            X_list.append(wave)
            cords_list.append([long, lat, elev])
            target_coords_list.append([target_long, target_lat])
            # y_list.append(label)

        X = np.stack(X_list, axis=0)  # (N_station, T, C)
        target_long, target_lat = np.mean(target_coords_list, axis=0)
        y = np.asarray([target_long, target_lat])
        y = (y - self.Y_MEAN) / self.Y_STD

        cords = np.asarray(cords_list, dtype=np.float32)

        x_out = [X, cords]
        return x_out, y

    # ------------------------------------------------
    def build_window(self):
        df = self.df

        P = df["p_arrival_sample"].to_numpy()
        S = df["s_arrival_sample"].to_numpy()
        event_idx = df["event_idx"].to_numpy()

        window_start = np.arange(
            0, self.max_length - self.window_size + 1, self.stride
        )
        window_end = window_start + self.window_size

        P_left = np.maximum(0, P - self.x_margin)
        P_right = np.minimum(self.max_length, P + self.x_margin)
        S_left = np.maximum(0, S - self.x_margin)
        S_right = np.minimum(self.max_length, S + self.x_margin)

        valid_P = (window_start[None, :] < P_right[:, None]) & (
                window_end[None, :] > P_left[:, None]
        )
        valid_S = (window_start[None, :] < S_right[:, None]) & (
                window_end[None, :] > S_left[:, None]
        )

        # Window valid jika ADA minimal 1 stasiun aktif
        valid_any = valid_P | valid_S

        row_idx, win_idx = np.where(valid_any)

        windows = np.stack(
            [
                event_idx[row_idx],
                window_start[win_idx],
            ],
            axis=1,
        )

        # unique event-window
        windows = np.unique(windows, axis=0)

        return windows