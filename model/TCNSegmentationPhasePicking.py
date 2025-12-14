import torch
from torch import nn
from .layers.SeparableConv import SeparableConvLayer
from .layers.TemporalConv import TemporalConvLayer, DMLDropout
from .loader import ModelLoader
from torch.functional import F
from .TCNSegmentation import EncoderBlock, DecoderBlock

@ModelLoader.register("TCNSegmentationPhasePicking")
class Models(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.__class__.__name__ = "TCNSegmentationPhasePicking"

        self.enc1 = EncoderBlock(3, 8, 9)
        self.enc2 = EncoderBlock(8, 16, 7)
        self.enc3 = EncoderBlock(16, 32, 5)
        self.enc4 = EncoderBlock(32, 64, 3, pool=True)

        # backbone
        layers = []
        for i in range(levels):
            dilation = 2**i
            layers.append(TemporalConvLayer(64, 5, dilation))

        self.network = nn.Sequential(*layers)

        # TODO: change from upsample to nn.ConvTranspose1d

        # only change the head
        self.head = nn.Sequential(
            DecoderBlock(64, 32, 3),
            DecoderBlock(32, 16, 5),
            DecoderBlock(16, 8, 7),
            DecoderBlock(8, 4, 5),
            nn.Conv1d(4, 1, kernel_size=1),
        )

    def forward(self, x):
        orig_len = x.size(-1)
        pad_to = ((orig_len + 15) // 16) * 16  # next multiple of 16
        pad_amount = pad_to - orig_len

        x = F.pad(x, (0, pad_amount))
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.network(x)

        x = self.head(x)
        return x[..., :orig_len]
