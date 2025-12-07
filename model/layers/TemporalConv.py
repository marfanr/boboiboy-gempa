from torch import nn
import torch
from .CausalConv import CausalConv1DLayer

class DMLDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Dropout layer yang kompatibel dengan DML GPU
        Args:
            p (float): probabilitas dropout
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("p harus di antara 0 dan 1")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # Buat mask langsung di device yang sama dengan x (DML GPU)
        mask = (torch.rand_like(x, device=x.device) > self.p).to(x.dtype) / (1 - self.p)
        return x * mask

class TemporalConvLayer(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1DLayer(ch, ch, k, dilation)
        self.conv2 = CausalConv1DLayer(ch, ch, k, dilation)
        self.do = DMLDropout(dropout)
        self.res = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        res = self.res(x)
        out = torch.relu(self.conv1(x))
        out = self.do(out)
        out = torch.relu(self.conv2(out))
        out = self.do(out)

        return torch.relu(out + res)
