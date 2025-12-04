from torch import nn

class CausalConv1DLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, dilation=1):
        super().__init__()
        self.pad = (ksize - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            dilation=dilation
        )
        
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)