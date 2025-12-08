from .layers.register import LAYER_REGISTRY

class ConfigParser:
    def __init__(self, cfg: str):
        self.file = cfg
        
        print("Registered layers:")
        for i in LAYER_REGISTRY:
            print(i)
        print("End of registered layers\n")
        
    
    def parse(self):
        layers = []
        with open(self.file, 'r') as f:
            lines = f.read().splitlines()

        layer_dict = {}
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if line.startswith('['):  # section baru
                if layer_dict:
                    layers.append(layer_dict)
                layer_dict = {'type': line[1:-1]}  # ambil nama section tanpa []
            else:
                key, value = line.split('=')
                layer_dict[(key.strip()).lower()] = value.strip()
        if layer_dict:
            layers.append(layer_dict)
            
        print(layers)
        for l in layers:
            type = l["type"]
            print(f"Parsing layer of type: {type}")
            if type == "net":
                batch = int(l.get("batch", 32))
                print(f"Batch size: {batch}")
                continue
            
            if type not in LAYER_REGISTRY:
                raise ValueError(f"Layer type {type} not registered")
            
            # (**{k: v for k, v in l.items() if k != "type"}
            layer = LAYER_REGISTRY[type]().build()
            print(f"Built layer: {layer}")
        # return layers