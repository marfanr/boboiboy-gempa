import torch
from torch import nn
from .layers.SeparableConv import SeparableConvLayer
from .layers.TemporalConv import TemporalConvLayer, DMLDropout
from .loader import ModelLoader
from torch.functional import F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pool=True):
        super().__init__()
        self.__class__.__name__ = "EncoderBlock"
        layers = [
            nn.Conv1d(in_ch, out_ch, ksize, padding=ksize // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            DMLDropout(0.3),
            nn.Conv1d(out_ch, out_ch, ksize, padding=ksize // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            DMLDropout(0.3),
        ]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.__class__.__name__ = "DecoderBlock"

        # --- Main path ---
        self.main = nn.Sequential(
            # upsample (in_ch -> out_ch)
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            # conv (out_ch -> out_ch)
            nn.Conv1d(
                out_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm1d(out_ch),
        )

        # --- Shortcut path ---
        shortcut_layers = [
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        ]

        # jika butuh penyesuaian channel (in_ch != out_ch)
        if in_ch != out_ch:
            shortcut_layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False))
            shortcut_layers.append(nn.BatchNorm1d(out_ch))

        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))


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
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.GLU(dim=1),  # input 6 → output 3
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
