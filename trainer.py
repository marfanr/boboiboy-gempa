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
import torch
from ignite.handlers import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.Writer import TensorWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        logger: TensorWriter = None,
        optimizer=None,
        criterion=None,
    ):
        print(f"iterasi_per_epoch {len(train)} , {len(test)}")
        self.train_dl = train
        self.test_dl = test
        self.model = model
        # TODO
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs")
        #     # model = DDP(
        #     #     model,
        #     # )

        model = model.to(device=device)

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optimizer = torch.optim.Adam(
                model.parameters(), lr=0.0001, foreach=False, weight_decay=1e-6
            )

        self.scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,  min_lr=1e-6
        )

        self.logger = logger
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
            f"[{timestamp}] Epoch {current_epoch} | Iter {current_iter} / {current_epoch * engine.state.epoch_length} - Loss: {loss:.4f}"
        )

    def eent_on_epoch_complete(self, engine):
        epoch = engine.state.epoch

        print(f"Model saved at {self.output} {epoch}.pth")
        print("1 epoch complete")
        self.evaluator.run(self.test_dl)
        self.scheduler.step()
        self.best_checkpointer(
            self.evaluator,
            {
                "model": self.model,
                "optimizer_state": self.optimizer,
                "scheduler_state": self.scheduler,
            },
        )
        metrics = self.evaluator.state.metrics
        print(
            f"Epoch {engine.state.epoch} - Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}"
        )
        print(f"scheduler state: {self.scheduler.state_dict()}")
        if self.logger is not None:
            self.logger.write_scalar("val", "loss", metrics["loss"], epoch)
            self.logger.write_scalar("val", "accuracy", metrics["accuracy"], epoch)

    def _prepare_batch(self, batch, device):
        x, y = batch
        return x.to(device), y.to(device)

    def _per_iterate_log(self, engine):
        if self.logger is not None:
            self.logger.write_scalar(
                "train", "loss", engine.state.output, engine.state.iteration
            )

    def train(self, max_epoch, weight, output, distributed=False):
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
            filename_prefix=f"best-{self.model.__class__.__name__}-{int(time.time())}",
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
                weight, map_location="cpu"
            )  # atau device lain
            self.model.load_state_dict(best_checkpoints["model"])
            self.model = self.model.to(device=device)
            if "optimizer_state" in best_checkpoints:
                self.optimizer.load_state_dict(best_checkpoints["optimizer_state"])
            if "scheduler_state" in best_checkpoints:
                self.scheduler.load_state_dict(best_checkpoints["scheduler_state"])

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
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self._per_iterate_log
        )
        self.trainer.run(self.train_dl, max_epochs=max_epoch)
