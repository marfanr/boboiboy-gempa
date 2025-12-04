
class ModelLoader:
    # Dictionary privat untuk menyimpan "ID": Class
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Ini adalah decorator untuk mendaftarkan class.
        """
        def inner_wrapper(wrapped_class):
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get(cls, name, *args, **kwargs):
        """
        Mengambil class berdasarkan ID dan langsung membuat object-nya.
        """
        if name not in cls._registry:
            raise ValueError(f"Class dengan ID '{name}' tidak ditemukan.")
        
        # Mengembalikan instance dari class tersebut
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def list_keys(cls):
        """Melihat semua ID yang terdaftar"""
        return list(cls._registry.keys())