LAYER_REGISTRY = {}

def register_layer(name):
    def decorator(cls):
        LAYER_REGISTRY[name] = cls
        return cls
    return decorator
