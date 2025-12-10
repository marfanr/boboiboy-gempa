import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional


class TensorWriter():
    def __init__(self, experiment_name: str, base_dir: str = "runs", note: str = None):
        """
        Tensor Writer
        """
        self.writers: Dict[str, SummaryWriter] = {}
        current_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.log_dir = os.path.join(base_dir, f"{experiment_name}_{current_time}")
        if note is not None:
            self.log_dir = os.path.join(self.log_dir, "_".join(note))
        print(f"Logging to {self.log_dir}")

    def get_writer(self, name):
        if name not in self.writers:
            writer_path = os.path.join(self.log_dir, name)
            self.writers[name] = SummaryWriter(log_dir=writer_path)
            return self.writers[name]
        return self.writers[name]

    def write_scalar(self, name: str, tag: str, value: float, step: Optional[int] = 1):
        w = self.get_writer(name)
        w.add_scalar(tag, value, step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
