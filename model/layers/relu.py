from .builder import LayerBuilder
import torch.nn as nn
from .register import register_layer


@register_layer("relu")
class ReLULayer(LayerBuilder):
    def __init__(self, block):
        super().__init__(block)

    def build(self):
        return nn.ReLU()
