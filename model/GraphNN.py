from torch import nn
import torch
from torch.functional import F
from .loader import ModelLoader
from torch_geometric.nn import GCNConv


@ModelLoader.register("GraphNN")
class GraphNN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_layer: int = 3):
        super().__init__()
        self.__class__.__name__ = "GraphNN"

    def forward(self, x):
        pass
