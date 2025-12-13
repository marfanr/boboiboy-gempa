import torch
from torch import nn
from torch.nn import functional as F
from .layers.TemporalConv import TemporalConvLayer, DMLDropout
from .loader import  ModelLoader

class ChannelAttention(nn.Module):
    """Channel attention untuk fokus pada fitur penting"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pool=True, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ksize, padding=ksize // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            DMLDropout(dropout),
            nn.Conv1d(out_ch, out_ch, ksize, padding=ksize // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            DMLDropout(dropout),
        )
        self.attention = ChannelAttention(out_ch)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
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
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            DMLDropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.attention = ChannelAttention(out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.attention(x)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling untuk multi-scale features"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=3, dilation=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=6, dilation=6),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=9, dilation=9),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
        ])

        self.project = nn.Sequential(
            nn.Conv1d(out_ch * 4, in_ch, 1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


@ModelLoader.register("TCNSegmentationNew")
class TCNSegmentation(nn.Module):
    def __init__(self, levels=6, dropout=0.2):
        super().__init__()
        self.__class__.__name__ = "TCNSegmentationNew"

        # Encoder
        self.enc1 = EncoderBlock(3, 16, 9, dropout=dropout)
        self.enc2 = EncoderBlock(16, 32, 7, dropout=dropout)
        self.enc3 = EncoderBlock(32, 64, 5, dropout=dropout)
        self.enc4 = EncoderBlock(64, 128, 3, pool=True, dropout=dropout)

        # Multi-scale feature extraction
        self.aspp = ASPP(128, 32)

        # TCN backbone
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalConvLayer(128, 5, dilation))
        self.network = nn.Sequential(*layers)

        # Decoder
        self.dec4 = DecoderBlock(128, 64, 3, dropout=dropout)
        self.dec3 = DecoderBlock(64, 32, 5, dropout=dropout)
        self.dec2 = DecoderBlock(32, 16, 7, dropout=dropout)
        self.dec1 = DecoderBlock(16, 8, 9, dropout=dropout)

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv1d(8, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, 1)
        )

    def forward(self, x):
        orig_len = x.size(-1)
        pad_to = ((orig_len + 15) // 16) * 16
        pad_amount = pad_to - orig_len

        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount))

        # Encoding
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # Multi-scale features + TCN
        x = self.aspp(x)
        x = self.network(x)

        # Decoding
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        # Segmentation
        x = self.head(x)

        return x[..., :orig_len]


