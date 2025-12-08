class LayerBuilder:
    def __init__(self, block):
        self.block = block

    def build(self, in_channels):
        raise NotImplementedError
