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
        x_margin=200,
        normalize=True,
        noise_level=0.01,
        windows=None,
        use_balancing: bool = False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
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
        self.use_balancing = use_balancing
        self.device = device
        self.h5 = None
        
        if windows is None:
            self.windows = self._build_windows()
        else:
            self.windows = windows
            
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
        
        # Convert to torch tensors and move to device
        P = torch.from_numpy(self.df.p_arrival_sample.to_numpy()).to(self.device)
        S = torch.from_numpy(self.df.s_arrival_sample.to_numpy()).to(self.device)
        
        # Compute margins using torch operations
        P_left = torch.clamp(P - self.x_margin, min=0)
        P_right = torch.clamp(P + self.x_margin, max=6000)
        S_left = torch.clamp(S - self.x_margin, min=0)
        S_right = torch.clamp(S + self.x_margin, max=6000)
        
        # Generate window starts and ends
        window_starts = torch.arange(0, 6000 - self.data_length + 1, self.stride, device=self.device)
        window_ends = window_starts + self.data_length
        
        # Compute validity masks using broadcasting
        # Shape: (num_samples, num_windows)
        P_valid = (window_starts.unsqueeze(0) < P_right.unsqueeze(1)) & \
                  (window_ends.unsqueeze(0) > P_left.unsqueeze(1))
        S_valid = (window_starts.unsqueeze(0) < S_right.unsqueeze(1)) & \
                  (window_ends.unsqueeze(0) > S_left.unsqueeze(1))
        
        # Check if arrivals are actually in windows
        P_in = (P.unsqueeze(1) >= window_starts.unsqueeze(0)) & \
               (P.unsqueeze(1) < window_ends.unsqueeze(0))
        S_in = (S.unsqueeze(1) >= window_starts.unsqueeze(0)) & \
               (S.unsqueeze(1) < window_ends.unsqueeze(0))
        
        P_eff = P_valid & P_in
        S_eff = S_valid & S_in
        
        # Count categories
        category = {
            "P_and_S": int((P_eff & S_eff).sum().item()),
            "P_only": int((P_eff & ~S_eff).sum().item()),
            "S_only": int((~P_eff & S_eff).sum().item()),
            "no_P_no_S": int(((P_valid | S_valid) & ~P_eff & ~S_eff).sum().item()),
        }
        print(category)
        
        # Get indices for each category
        idx_p_and_s = torch.nonzero(P_eff & S_eff, as_tuple=False)
        idx_p_only = torch.nonzero(P_eff & ~S_eff, as_tuple=False)
        idx_s_only = torch.nonzero(~P_eff & S_eff, as_tuple=False)
        idx_no_p_no_s = torch.nonzero((P_valid | S_valid) & ~P_eff & ~S_eff, as_tuple=False)
        
        # Convert to list of tuples and move to CPU
        windows_p_and_s = [(int(i[0]), int(i[1])) for i in idx_p_and_s.cpu()]
        windows_p_only = [(int(i[0]), int(i[1])) for i in idx_p_only.cpu()]
        windows_s_only = [(int(i[0]), int(i[1])) for i in idx_s_only.cpu()]
        windows_no_p_no_s = [(int(i[0]), int(i[1])) for i in idx_no_p_no_s.cpu()]
        
        windows = (
            windows_p_and_s + 
            windows_p_only + 
            windows_s_only + 
            windows_no_p_no_s
        )
        
        # Shuffle using torch
        indices = torch.randperm(len(windows))
        windows = [windows[i] for i in indices.tolist()]
        
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
            label[int(a):int(b)] = 1.0
        
        return x_window, label.unsqueeze(0)
    
    def _normalize(self, wave):
        mean = wave.mean(dim=1, keepdim=True)
        std = wave.std(dim=1, keepdim=True) + 1e-6
        return (wave - mean) / std
    
    def __augmentation__(self, x_window):
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x_window) * self.noise_level
            x_window.add_(noise)
        return x_window
    
    def _balancing(self):
        pass