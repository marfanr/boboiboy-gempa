import torch
from torch import nn
from .layers.SeparableConv import SeparableConvLayer
from .layers.TemporalConv import TemporalConvLayer
from .loader import ModelLoader
from torch.functional import F


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            SeparableConvLayer(3, 8, kernel_size=9, padding=4),
            nn.MaxPool1d(2),
            SeparableConvLayer(8, 16, kernel_size=7, padding=3),
            nn.MaxPool1d(2),
            SeparableConvLayer(16, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            SeparableConvLayer(32, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            # 1. Perbesar ukuran sinyal (misal: dari 63 -> 126)
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            # 2. Olah fiturnya
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
        )

        layers = [nn.Upsample(scale_factor=2, mode="linear", align_corners=False)]

        # Jika channel berubah, tambahkan 1x1 Conv untuk menyesuaikan channel
        if in_ch != out_ch:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))

        self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        skip = self.shortcut(skip)
        return torch.relu(x + skip)


@ModelLoader.register("TCNSegmentation")
class Models(nn.Module):
    def __init__(self, levels):
        super().__init__()

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
        self.dec4 = DecoderBlock(64, 32, 3)  # skip from enc4
        self.dec3 = DecoderBlock(32, 16, 5)  # skip from enc3
        self.dec2 = DecoderBlock(16, 8, 7)  # skip from enc2
        self.dec1 = DecoderBlock(8, 3, 9)  # skip from enc1

        self.head = nn.Conv1d(3, 1, 1)

    def forward(self, x):
        orig_len = x.size(-1)
        pad_to = ((orig_len + 15) // 16) * 16  # next multiple of 16
        pad_amount = pad_to - orig_len

        x = F.pad(x, (0, pad_amount))
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x = self.network(x)

        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = self.head(x)
        return x[..., :orig_len]
