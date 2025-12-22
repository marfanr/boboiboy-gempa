import torch
from torch import nn
from torch.nn import functional as F
from .layers.TemporalConv import TemporalConvLayer, DMLDropout
from .loader import ModelLoader

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
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
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


class SpatialDropout1d(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        # x: (B, C, T) -> (B, C, 1, T)
        x = x.unsqueeze(2)
        x = self.dropout(x)
        return x.squeeze(2)


class ResCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()

        padding = kernel_size // 2 if kernel_size == 3 else 0

        self.dropout = SpatialDropout1d(dropout)

        self.norm1 = nn.BatchNorm1d(channels, eps=0.001, momentum=0.1)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=padding
        )

        self.norm2 = nn.BatchNorm1d(channels, eps=0.001, momentum=0.1)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=padding
        )

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = self.dropout(out)

        if out.size(-1) != identity.size(-1):
            identity = identity[..., : out.size(-1)]

        return out + identity


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
    def __init__(self, levels=4, dropout=0.2):
        super().__init__()
        self.__class__.__name__ = "TCNSegmentationV2"

        # ---------- Encoder ----------
        self.enc = nn.ModuleList(
            [
                EncoderBlock(3, 8, 11, dropout=dropout),
                EncoderBlock(8, 16, 9, dropout=dropout),
                EncoderBlock(16, 16, 9, dropout=dropout),
                EncoderBlock(16, 32, 7, dropout=dropout),
                EncoderBlock(32, 32, 7, dropout=dropout),
            ]
        )

        # ---------- ResCNN ----------
        self.res_cnn = nn.ModuleList(
            [
                ResCNNBlock(32, kernel_size=3, dropout=dropout),
                ResCNNBlock(32, kernel_size=3, dropout=dropout),
                ResCNNBlock(32, kernel_size=3, dropout=dropout),
            ]
        )

        # ---------- Transformer (Bottleneck) ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=8,
            dim_feedforward=256,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=True,  # stabil untuk deep model
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # ---------- TCN ----------
        tcn_layers = []
        for i in range(levels):
            dilation = 2**i
            tcn_layers.append(TemporalConvLayer(ch=32, k=5, dilation=dilation))
        self.tcn = nn.Sequential(*tcn_layers)

        # ---------- Decoder ----------
        self.dec = nn.ModuleList(
            [
                DecoderBlock(32, 32, 7, dropout=dropout),
                DecoderBlock(32, 16, 7, dropout=dropout),
                DecoderBlock(16, 16, 9, dropout=dropout),
                DecoderBlock(16, 8, 11, dropout=dropout),
                DecoderBlock(8, 8, 11, dropout=dropout),
            ]
        )

        # ---------- Segmentation Head ----------
        self.head = nn.Sequential(
            DynamicLayerNorm(eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, 1),
        )

    def forward(self, x):
        orig_len = x.size(-1)

        # Encoder
        for block in self.enc:
            x = block(x)

        # ResCNN
        for block in self.res_cnn:
            x = block(x)

        x = self.tcn(x)

        # ---------- Transformer ----------
        # (B, C, T) -> (T, B, C)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # back to (B, C, T)

        # ---------- TCN ----------

        # Decoder
        for block in self.dec:
            x = block(x)

        # Head
        x = self.head(x)

        # Restore original temporal resolution
        return F.interpolate(x, size=orig_len, mode="linear", align_corners=False)
