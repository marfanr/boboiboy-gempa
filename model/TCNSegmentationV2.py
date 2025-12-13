import torch
from torch import nn
from torch.nn import functional as F
from .layers.TemporalConv import TemporalConvLayer, DMLDropout
from .loader import  ModelLoader

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pool=True, dropout=0.2):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, ksize, padding=ksize // 2),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.ModuleList(layers)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True) if pool else nn.Identity()

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = self.pool(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()

        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch,
            kernel_size=4, stride=2, padding=1
        )

        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        # x = self.attention(x)
        return x

class DynamicLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.ln = None

    def forward(self, x):
        if self.ln is None:
            normalized_shape = x.shape[1:]
            self.ln = nn.LayerNorm(normalized_shape, eps=self.eps).to(x.device)
        return self.ln(x)


@ModelLoader.register("TCNSegmentationV2")
class TCNSegmentationV2(nn.Module):
    def __init__(self, levels=3, dropout=0.2):
        super().__init__()
        self.__class__.__name__ = "TCNSegmentationV2"

        # ---------- Encoder ----------
        self.enc1 = EncoderBlock(3, 8, 11, dropout=dropout)
        self.enc2 = EncoderBlock(8, 16, 9, dropout=dropout)
        self.enc3 = EncoderBlock(16, 32, 7, dropout=dropout)
        self.enc4 = EncoderBlock(32, 64, 5, pool=True, dropout=dropout)

        # ---------- TCN ----------
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            layers.append(
                TemporalConvLayer(
                    ch=64,
                    k=5,
                    dilation=dilation
                )
            )
        self.network = nn.Sequential(*layers)

        # ---------- Decoder ----------
        self.dec4 = DecoderBlock(64, 32, 5, dropout=dropout)
        self.dec3 = DecoderBlock(32, 16, 7, dropout=dropout)
        self.dec2 = DecoderBlock(16, 8, 9, dropout=dropout)
        self.dec1 = DecoderBlock(8, 4, 11, dropout=dropout)

        # ---------- Segmentation Head ----------
        self.head = nn.Sequential(
            nn.Conv1d(4, 8, 3, padding=1),
            # nn.LayerNorm(1200),
            DynamicLayerNorm(eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, 1)
        )

    def forward(self, x):
        orig_len = x.size(-1)

        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # TCN
        x = self.network(x)

        # Decoder
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        # Head
        x = self.head(x)

        return F.interpolate(x, orig_len, mode="linear", align_corners=False)
