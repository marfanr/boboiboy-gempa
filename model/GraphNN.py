from torch import nn
import torch
from torch.functional import F


class GraphNN(nn.Module):
    def __init__(self):
        super().__init__()
