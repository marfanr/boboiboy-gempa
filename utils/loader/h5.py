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
            count,
            stride=500,
            offset_pos=0,
            x_margin=200,
            normalize=True,
            noise_level=0.4,
            windows=None,
            use_balancing: bool = True
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
            "min": 0.7,  # kelas paling kecil
            "2nd_min": 0.8,  # kelas kedua paling kecil
            "middle": 0.75,  # semua kelas di tengah
            "max": 0.7  # kelas terbesar (undersample)
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
        P_valid = (window_starts[None, :] < P_right[:, None]) & \
                  (window_ends[None, :] > P_left[:, None])

        S_valid = (window_starts[None, :] < S_right[:, None]) & \
                  (window_ends[None, :] > S_left[:, None])

        # ---------------------------
        # 3. Window yang benar-benar mengandung arrival (lebih ketat)
        # ---------------------------
        P_in = (P[:, None] >= window_starts[None, :]) & (P[:, None] < window_ends[None, :])
        S_in = (S[:, None] >= window_starts[None, :]) & (S[:, None] < window_ends[None, :])

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
                self.windows_p_and_s +
                self.windows_p_only +
                self.windows_s_only +
                self.windows_no_p_no_s
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
            label[int(a): int(b)] = 1.0

        return x_window, label.unsqueeze(0)

    def _normalize(self, wave):
        mean = wave.mean(dim=1, keepdim=True)
        std = wave.std(dim=1, keepdim=True) + 1e-6
        return (wave - mean) / std

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
            print(f"  {class_name}: {n_samples:,} -> {target:,} (oversampled x{target / n_samples:.2f})")
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
        balanced_p_and_s = self.__balance_samples(self.windows_p_and_s, targets["P_and_S"], "P_and_S")
        balanced_p_only = self.__balance_samples(self.windows_p_only, targets["P_only"], "P_only")
        balanced_s_only = self.__balance_samples(self.windows_s_only, targets["S_only"], "S_only")
        balanced_no_p_no_s = self.__balance_samples(self.windows_no_p_no_s, targets["no_P_no_S"], "no_P_no_S")

        balanced_windows = (
                balanced_p_and_s +
                balanced_p_only +
                balanced_s_only +
                balanced_no_p_no_s
        )

        # 4. Shuffle agar tidak berurutan per kategori
        np.random.shuffle(balanced_windows)

        # 5. Buat label untuk setiap window
        # 0: P_and_S, 1: P_only, 2: S_only, 3: no_P_no_S
        balanced_labels = (
                [0] * len(balanced_p_and_s) +
                [1] * len(balanced_p_only) +
                [2] * len(balanced_s_only) +
                [3] * len(balanced_no_p_no_s)
        )

        # Shuffle labels sesuai dengan shuffle windows
        indices = list(range(len(balanced_windows)))
        np.random.shuffle(indices)
        balanced_windows = [balanced_windows[i] for i in indices]
        balanced_labels = [balanced_labels[i] for i in indices]

        print("\n" + "=" * 50)
        print("HASIL SETELAH BALANCING:")
        print(f"Total windows: {len(balanced_windows):,}")
        print(f"  P_and_S: {balanced_labels.count(0):,}")
        print(f"  P_only: {balanced_labels.count(1):,}")
        print(f"  S_only: {balanced_labels.count(2):,}")
        print(f"  no_P_no_S: {balanced_labels.count(3):,}")
        print("=" * 50)

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
