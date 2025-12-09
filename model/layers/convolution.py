from .builder import LayerBuilder
import torch.nn as nn
from .register import register_layer


@register_layer("convolutional")
class ConvolutionLayer(LayerBuilder):
    def __init__(self, block):
        super().__init__(block)

        self.in_channels = int(block.get("in_channels", 1))
        self.out_channels = int(block.get("filters", 1))
        self.kernel_size = int(block.get("size", 1))
        self.stride = int(block.get("stride", 1))
        self.padding = int(block.get("pad", 0))


    def build(self):
        return nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
