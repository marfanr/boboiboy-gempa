from .layers.register import LAYER_REGISTRY
from torch import nn


class NetBuilder(nn.Module):
    def __init__(self):
        pass


class ConfigParser:
    def __init__(self, cfg: str):
        self.file = cfg
        self.builded_layers = []
        self.routed_map = []
        self.module_list = []

        print("Registered layers:")
        for i in LAYER_REGISTRY:
            print(i)
        print("End of registered layers\n")

    def __parse_cfg(self) -> list:
        layers = []
        with open(self.file, "r") as f:
            lines = f.read().splitlines()

        layer_dict = {}
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("["):  # section baru
                if layer_dict:
                    layers.append(layer_dict)
                layer_dict = {"type": line[1:-1]}  # ambil nama section tanpa []
            else:
                key, value = line.split("=")
                layer_dict[(key.strip()).lower()] = value.strip()
        if layer_dict:
            layers.append(layer_dict)

        return layers

    def parse(self):
        layers = self.__parse_cfg()

        for (
                l,
                i,
        ) in zip(layers, range(len(layers))):

            type = l["type"]
            print(f"Parsing layer of type: {type}")
            if type == "net":
                batch = int(l.get("batch", 32))
                print(f"Batch size: {batch}")
                continue

            if type == "route":
                layers_idxs = l.get("layers", "")
                print(f"Route layers: {layers_idxs}")
                last_parent = [
                    i + int(x) if int(x) < 0 else int(x) for x in layers_idxs.split(",")
                ]
                continue

            if type not in LAYER_REGISTRY:
                raise ValueError(f"Layer type {type} not registered")

            params = {k: v for k, v in l.items() if k != "type"}
            print(f"Parameters for layer {type}: {params}")

            layer = LAYER_REGISTRY[type](params).build()
            activation_name = l.get("activation", "linear")

            activation_dict = nn.ModuleDict({
                "relu": nn.ReLU(),
            })
            activate = LAYER_REGISTRY[activation_name](params).build()
            print(f"Built layer: {layer} activation: {activate}")

            self.builded_layers.append(layer)

            # (**{k: v for k, v in l.items() if k != "type"}
            # layer = LAYER_REGISTRY[type]().build()
            # print(f"Built layer: {layer}")
        # return layers
