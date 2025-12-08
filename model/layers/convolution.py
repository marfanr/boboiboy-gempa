from .builder import LayerBuilder
import torch.nn as nn
from .register import register_layer


@register_layer("convolutional")
class ConvolutionLayer(LayerBuilder):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self):
        return nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
