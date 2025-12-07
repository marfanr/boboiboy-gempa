class ConfigParser:
    def __init__(self, cfg: str):
        self.file = cfg
        pass
    
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
                layer_dict[key.strip()] = value.strip()
        if layer_dict:
            layers.append(layer_dict)
        return layers