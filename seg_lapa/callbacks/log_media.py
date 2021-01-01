from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn

# Specific to logging media to disk
import cv2
import math
from pathlib import Path
from seg_lapa.utils.segmentation_label2rgb import LabelToRGB, Palette


class Mode(Enum):
    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


@dataclass
class PredData:
    """Holds the data read and converted from the LightningModule's LogMediaQueue"""

    inputs: np.ndarray
    labels: np.ndarray
    preds: np.ndarray


class LogMediaQueue:
    """Holds a circular queue for each of train/val/test modes, each of which contain the latest N batches of data"""

    def __init__(self, max_len: int = 3):
        if max_len < 1:
            raise ValueError(f"Queue must be length >= 1. Given: {max_len}")

        self.max_len = max_len
        self.log_media = {
            Mode.TRAIN: deque(maxlen=self.max_len),
            Mode.VAL: deque(maxlen=self.max_len),
            Mode.TEST: deque(maxlen=self.max_len),
        }

    def clear(self):
        """Clear all queues"""
        for mode, queue in self.log_media.items():
            queue.clear()

    def append(self, data: Any, mode: Mode):
        """Add a batch of data to a queue. Mode selects train/val/test queue"""
        self.log_media[mode].append(data)

    def fetch(self, mode: Mode) -> List[Any]:
        """Fetch all the batches available in a queue. Empties the selected queue"""
        data_r = []
        while len(self.log_media[mode]) > 0:
            data_r.append(self.log_media[mode].popleft())

        return data_r

    def len(self, mode: Mode) -> int:
        """Get the number of elements in a queue"""
        return len(self.log_media[mode])


class LogMedia(Callback):
    """Logs model output images and other media to weights and biases

    This callback required adding an attribute to the LightningModule called ``self.log_media``. This is a cicular
    queue that holds the latest N batches. This callback fetches the latest data from the queue for logging.

    Use ``get_empty_data_queue()`` to get the data structure.

    Usage:
        import pytorch_lightning as pl

        class MyModel(pl.LightningModule):
            self.log_media: LogMediaQueue = LogMediaQueue(max_len)

        trainer = pl.Trainer(callbacks=[LogMedia()])

    Args:
        period_epoch (int): If > 0, log every N epochs
        period_step (int): If > 0, log every N steps (i.e. batches)
        max_samples (int): Max number of images to log
        save_to_disk (bool): If True, save results to disk
        logs_dir (str or Path): Path to directory where results will be saved
    """

    SUPPORTED_LOGGERS = [pl_loggers.WandbLogger]

    def __init__(
        self,
        max_samples: int = 10,
        period_epoch: int = 1,
        period_step: int = 0,
        save_to_disk: bool = True,
        logs_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.max_samples = max_samples
        self.period_epoch = period_epoch
        self.period_step = period_step
        self.save_to_disk = save_to_disk
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.valid_logger = False

        # Project-specific fields
        self.class_labels_lapa = {
            0: "background",
            1: "skin",
            2: "eyebrow_left",
            3: "eyebrow_right",
            4: "eye_left",
            5: "eye_right",
            6: "nose",
            7: "lip_upper",
            8: "inner_mouth",
            9: "lip_lower",
            10: "hair",
        }

    def setup(self, trainer, pl_module, stage: str):
        # This callback requires a ``.log_media`` attribute in LightningModule
        req_attr = "log_media"
        if not hasattr(pl_module, req_attr):
            raise AttributeError(
                f"{pl_module.__class__.__name__}.{req_attr} not found. The {LogMedia.__name__} "
                f"callback requires the LightningModule to have the {req_attr} attribute."
            )
        if not isinstance(pl_module.log_media, LogMediaQueue):
            raise AttributeError(f"{pl_module.__class__.__name__}.{req_attr} must be of type {LogMediaQueue.__name__}")

        if self.verbose:
            pl_module.print(f"Initializing Callback {LogMedia.__name__}")

        # TODO: Create log dir within callback.
        if trainer.is_global_zero:
            if self.save_to_disk:
                if self.logs_dir is None:
                    raise ValueError(
                        f"Callback {LogMedia.__name__}: Invalid logs_dir: {self.logs_dir}. Please give "
                        f"valid path for logs_dir"
                    )
                # else:
                #     self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.valid_logger = True if self._logger_is_supported(trainer) else False

        # TODO: Train and test queues are still empty?

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.TRAIN)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.VAL)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.TEST)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if not self._should_log_epoch(trainer):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.TRAIN)
        self._save_results_to_disk(pred_data, Mode.TRAIN)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.VAL)
        self._save_results_to_disk(pred_data, Mode.VAL)

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer):
            return

        pred_data = self._get_preds_from_lightningmodule(pl_module, Mode.TRAIN)
        self._log_images_to_wandb(trainer, pred_data, Mode.TEST)
        self._save_results_to_disk(pred_data, Mode.TEST)

    def _should_log_epoch(self, trainer):
        if trainer.running_sanity_check:
            return False
        if self.period_epoch < 1 or ((trainer.current_epoch + 1) % self.period_epoch != 0):
            return False
        return True

    def _should_log_step(self, trainer, batch_idx):
        if trainer.running_sanity_check:
            return False
        if self.period_step < 1 or ((batch_idx + 1) % self.period_step != 0):
            return False
        return True

    @rank_zero_only
    def _logger_is_supported(self, trainer):
        """This callback only works with wandb logger"""
        for logger_type in self.SUPPORTED_LOGGERS:
            if isinstance(trainer.logger, logger_type):
                return True

        rank_zero_warn(
            f"WARN: Unsupported logger, will not log any media to logger this run."
            f" Supported loggers: {[sup_log.__name__ for sup_log in self.SUPPORTED_LOGGERS]}. Given: {trainer.logger}."
        )
        return False

    @rank_zero_only
    def _get_preds_from_lightningmodule(self, pl_module, mode: Mode) -> Optional[PredData]:
        """Fetch latest N batches from the data queue in LightningModule.
        Process the tensors as required (example, convert to numpy arrays and scale)
        """
        if pl_module.log_media.len(mode) == 0:  # Empty queue
            return None

        media_data = pl_module.log_media.fetch(mode)

        inputs = torch.cat([x["inputs"] for x in media_data], dim=0)
        labels = torch.cat([x["labels"] for x in media_data], dim=0)
        preds = torch.cat([x["preds"] for x in media_data], dim=0)

        # Limit the num of samples and convert to numpy
        inputs = inputs[: self.max_samples].detach().cpu().numpy().transpose((0, 2, 3, 1))
        inputs = (inputs * 255).astype(np.uint8)
        labels = labels[: self.max_samples].detach().cpu().numpy().astype(np.uint8)
        preds = preds[: self.max_samples].detach().cpu().numpy().astype(np.uint8)

        out = PredData(inputs=inputs, labels=labels, preds=preds)

        return out

    @rank_zero_only
    def _save_results_to_disk(self, pred_data: Optional[PredData], mode: Mode):
        """For a given mode (train/val/test), save the results to disk"""
        if not self.save_to_disk:
            return
        if pred_data is None:  # Empty queue
            rank_zero_warn(f"Empty queue! Mode: {mode}")
            return

        # Get the latest batches from the data queue in LightningModule
        inputs, labels, preds = pred_data.inputs, pred_data.labels, pred_data.preds

        # Colorize labels and predictions
        label2rgb = LabelToRGB()
        labels_rgb = [label2rgb.map_color_palette(lbl, Palette.LAPA) for lbl in labels]
        preds_rgb = [label2rgb.map_color_palette(pred, Palette.LAPA) for pred in preds]
        inputs_l = [ipt for ipt in inputs]

        # Create collage of results
        results_l = []
        for inp, lbl, pred in zip(inputs_l, labels_rgb, preds_rgb):
            # Combine each pair of inp/lbl/pred into singe image
            res_combined = np.concatenate((inp, lbl, pred), axis=1)
            results_l.append(res_combined)
        # Create grid
        n_imgs = len(results_l)
        n_cols = 4  # Fix num of columns
        n_rows = int(math.ceil(n_imgs / n_cols))
        img_h, img_w, _ = results_l[0].shape
        grid_results = np.zeros((img_h * n_rows, img_w * n_cols, 3), dtype=np.uint8)
        for idy in range(n_rows):
            for idx in range(n_cols):
                grid_results[idy * img_h : (idy + 1) * img_h, idx * img_w : (idx + 1) * img_w, :] = results_l[idx + idy]

        # Save collage
        fname = Path(self.logs_dir) / f"results.{mode.name.lower()}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(grid_results, cv2.COLOR_RGB2BGR))

    @rank_zero_only
    def _log_images_to_wandb(self, trainer, pred_data: Optional[PredData], mode: Mode):
        """Log images to wandb at the end of a batch. Steps are common for train/val/test"""
        if not self.valid_logger:
            return
        if pred_data is None:  # Empty queue
            return

        # Get the latest batches from the data queue in LightningModule
        inputs, labels, preds = pred_data.inputs, pred_data.labels, pred_data.preds

        # Create wandb Image for logging
        mask_list = []
        for img, lbl, pred in zip(inputs, labels, preds):
            mask_img = wandb.Image(
                img,
                masks={
                    "predictions": {"mask_data": pred, "class_labels": self.class_labels_lapa},
                    "groud_truth": {"mask_data": lbl, "class_labels": self.class_labels_lapa},
                },
            )
            mask_list.append(mask_img)

        wandb_log_label = f"{mode.name.title()}/Predictions"
        trainer.logger.experiment.log({wandb_log_label: mask_list}, commit=False)
