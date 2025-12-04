from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from torch import nn
from ignite.engine import Engine
import torch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events
from ignite.handlers import EarlyStopping
import time
import torch_directml
import torch
from ignite.handlers import ModelCheckpoint

try:
    # 1. Prioritas Pertama: Cek CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: cuda (NVIDIA)")

    # 2. Prioritas Kedua: Cek DirectML (AMD/Intel/Windows)
    else:
        # Import di dalam blok ini agar tidak error jika user tidak punya library-nya
        import torch_directml

        device = torch_directml.device()
        print(f"Device: {device} (DirectML)")

except ImportError:
    # Jika library torch_directml tidak terinstall dan CUDA tidak ada
    device = torch.device("cpu")
    print("DirectML tidak terinstall. Menggunakan CPU.")

except Exception as e:
    # Fallback terakhir ke CPU jika terjadi error lain
    print(f"Tidak ada GPU CUDA atau DirectML yang terdeteksi ({e}).")
    device = torch.device("cpu")
    print("Menggunakan Device: cpu")


class Trainer:
    def __init__(
        self,
        train: DataLoader,
        test: DataLoader,
        model: nn.Module,
        optimizer=None,
        criterion=None,
    ):
        self.train_dl = train
        self.test_dl = test
        self.model = model

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optimizer = torch.optim.SGD(
                model.parameters(), lr=0.0001, momentum=0.9
            )

        # optimizer.load_state_dict(best_checkpoints['optimizer'])
        self.criterion = criterion
        if self.criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, engine, batch):
        self.model.train()
        inputs, targets = batch
        # print(inputs.shape)
        self.optimizer.zero_grad()
        device = next(self.model.parameters()).device

        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs, targets.to(device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def threshold_output(self, output):
        y_pred, y = output

        # from logits -> probability
        y_pred = torch.sigmoid(y_pred)

        # convert to 0/1 label
        y_pred = (y_pred > 0.5).long()

        return y_pred.view(-1), y.view(-1)

    # handler
    def score_function(self, engine):
        return -engine.state.metrics["loss"]

    def log_training_loss(self, engine):
        current_iter = engine.state.iteration
        current_epoch = engine.state.epoch
        loss = engine.state.output
        timestamp = time.strftime("%H:%M:%S", time.localtime())

        print(
            f"[{timestamp}] Epoch {current_epoch} | Iter {current_iter} - Loss: {loss:.4f}"
        )

    def eent_on_epoch_complete(self, engine):
        epoch = engine.state.epoch
        loss = engine.state.output

        EPS = 1e-8

        print(f"Model saved: {self.out}.pth at {epoch}.pth")
        print("1 epoch complete")
        self.evaluator.run(self.test_dl)
        self.best_checkpointer(
            self.evaluator, {"model": self.model, "optimizer": self.optimizer}
        )

    def train(self, max_epoch, weight, output):
        self.trainer = Engine(self.train_step)
        self.evaluator = create_supervised_evaluator(
            self.model,
            metrics={
                "accuracy": Accuracy(output_transform=self.threshold_output),
                "loss": Loss(
                    self.criterion,
                ),
            },
            device=device,
        )
        self.output = output
        if self.output is None:
            self.output = "checkpoints"

        self.best_checkpointer = ModelCheckpoint(
            filename_prefix="best",
            dirname=self.output,
            n_saved=1,  # simpan 1 best
            score_function=lambda engine: -engine.state.metrics[
                "loss"
            ],  # kalau loss turun dianggap lebih baik
            score_name="val_loss",
            create_dir=True,
            require_empty=False,
        )

        if weight is not None:
            best_checkpoints = torch.load(
                weight, map_location=device
            )  # atau device lain
            self.model.load_state_dict(best_checkpoints["model"])

        early_stopping = EarlyStopping(
            patience=5, score_function=self.score_function, trainer=self.trainer
        )
        self.evaluator.add_event_handler(Events.COMPLETED, early_stopping)
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=100), self.log_training_loss
        )
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.eent_on_epoch_complete
        )

        self.trainer.run(self.train_dl, max_epochs=max_epoch)
