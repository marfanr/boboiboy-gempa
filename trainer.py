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
from utility.Writer import TensorWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.functional import F


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice Loss untuk boundary detection yang lebih baik"""

    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return focal + dice


class Trainer:
    def __init__(
        self,
        train: DataLoader,
        test: DataLoader,
        model: nn.Module,
        logger: TensorWriter = None,
        optimizer=None,
        criterion=None,
        device=None,  # ← Tambahkan parameter device
    ):
        print(f"iterasi_per_epoch {len(train)} , {len(test)}")
        self.train_dl = train
        self.test_dl = test

        # Set device dengan prioritas: parameter -> CUDA -> DirectML -> CPU
        if device is None:
            self.device = self._get_best_device()
        else:
            self.device = device

        print(f"Training akan menggunakan device: {self.device}")

        # Move model to device
        self.model: nn.Module = model.to(self.device)

        if not self.is_directml_device(self.device):
            print(f"compiling model ...")
            self.model = torch.compile(self.model, mode="default")

            print(f"compiling model done")

        # Setup optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                foreach=False if self.is_directml_device(self.device) else True,
                weight_decay=1e-4,
            )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        self.logger = logger

        # Setup criterion
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = FocalDiceLoss()

    def is_directml_device(self, device):
        """Cek apakah device adalah DirectML"""
        device_str = str(device)
        return "privateuseone" in device_str or "dml" in device_str.lower()

    def _get_best_device(self):
        """
        Deteksi device terbaik dengan prioritas:
        1. CUDA (NVIDIA GPU)
        2. DirectML (AMD/Intel GPU di Windows)
        3. CPU
        """
        try:
            # Prioritas 1: CUDA
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"✓ CUDA tersedia: {torch.cuda.get_device_name(0)}")
                print(f"  - CUDA device count: {torch.cuda.device_count()}")
                return device

            # Prioritas 2: DirectML
            try:
                import torch_directml

                device = torch_directml.device()
                print(f"✓ DirectML tersedia: {device}")
                return device
            except ImportError:
                print("✗ DirectML tidak terinstall")
            except Exception as e:
                print(f"✗ DirectML error: {e}")

            # Prioritas 3: CPU (fallback)
            print("❗Menggunakan CPU (tidak ada GPU tersedia)")
            return torch.device("cpu")

        except Exception as e:
            print(f"❗ Error saat deteksi device: {e}")
            print("❗ Fallback ke CPU")
            return torch.device("cpu")

    def train_step(self, engine, batch):
        self.model.train()
        inputs, targets = batch

        self.optimizer.zero_grad(set_to_none=True)

        # Gunakan self.device
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(self.device))
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Batch loss invalid")
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
        metrics = self.evaluator.state.metrics
        self.scheduler.step(metrics=metrics["loss"])
        self.best_checkpointer(
            self.evaluator,
            {
                "model": self.model,
                "optimizer_state": self.optimizer,
                "scheduler_state": self.scheduler,
            },
        )
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
                "loss": Loss(self.criterion),
            },
            device=self.device,  # ← Gunakan self.device
        )
        self.output = output
        if self.output is None:
            self.output = "checkpoints"
        print(f"output : {self.output}")

        self.best_checkpointer = ModelCheckpoint(
            filename_prefix=f"best-{self.model.__class__.__name__}-{int(time.time())}",
            dirname=self.output,
            n_saved=1,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="val_loss",
            create_dir=True,
            require_empty=False,
        )

        if weight is not None:
            best_checkpoints = torch.load(weight, map_location="cpu", mmap=True)
            self.model.load_state_dict(best_checkpoints["model"])
            self.model = self.model.to(self.device)
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
