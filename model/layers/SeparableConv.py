from torch import nn

class SeparableConvLayer(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        layers = []

        if in_ch != out_ch:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))

        layers.append(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=out_ch,  # depthwise
            )
        )
        layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        # 3) Pointwise conv normal (channel mixing)
        layers.append(
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
