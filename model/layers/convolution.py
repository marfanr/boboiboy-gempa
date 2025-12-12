from .builder import LayerBuilder
import torch.nn as nn
from .register import register_layer


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, use_bn=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else None
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return self.act(x)


@register_layer("convolutional")
class ConvolutionLayer(LayerBuilder):
    def __init__(self, block):
        super().__init__(block)

        self.in_channels = int(block.get("in_channels", 1))
        self.out_channels = int(block.get("filters", 1))
        self.kernel_size = int(block.get("size", 1))
        self.stride = int(block.get("stride", 1))
        self.padding = int(block.get("pad", 0))
        self.batch_norm = int(block.get("batch_normalize", 0))

    def build(self):
        return ConvBlock(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                         self.batch_norm)


__all__ = ["ConvolutionLayer"]
