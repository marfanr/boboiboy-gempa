from torch import nn
import torch
from .CausalConv import CausalConv1DLayer


class TemporalConvLayer(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1DLayer(ch, ch, k, dilation)
        self.conv2 = CausalConv1DLayer(ch, ch, k, dilation)
        self.do = nn.Dropout(dropout)
        self.res = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        res = self.res(x)
        out = torch.relu(self.conv1(x))
        out = self.do(out)
        out = torch.relu(self.conv2(out))
        out = self.do(out)

        return torch.relu(out + res)
