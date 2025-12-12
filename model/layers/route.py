from .builder import LayerBuilder
from .register import register_layer
import torch
import torch.nn as nn

@register_layer("route")
class RouteLayer(LayerBuilder):
    def __init__(self, block):
        super().__init__(block)

        # parsing: "layers = -1, -3" → [ -1, -3 ]
        raw = block.get("layers", "")
        self.route_layers = [int(x) for x in raw.split(",")]

    def build(self):
        print(f"Building route layer: {self.route_layers}")
        return RouteOp(self.route_layers)


class RouteOp(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, *xs):
        # YOLO style: concatenation
        print(xs)
        return torch.cat(xs, dim=self.axis)
