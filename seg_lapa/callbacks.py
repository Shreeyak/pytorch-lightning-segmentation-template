from collections import deque
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import early_stopping, Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.distributed import rank_zero_only

# Specific to logging media to disk
import cv2
import math
from pathlib import Path
from seg_lapa.utils.segmentation_label2rgb import LabelToRGB, Palette


class Mode(Enum):
    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


class EarlyStopping(early_stopping.EarlyStopping):
    """Direct sub-class of PL's early-stopping, changing the default parameters to suit our project

    Args:
        monitor: Monitor a key validation metric (eg: IoU). Monitoring loss is not a good idea as it is an unreliable
                 indicator of model performance. Two models might have the same loss but different performance
                 or the loss might start increasing, even though performance does not decrease.

        min_delta: Project-dependent - choose a value for your metric below which you'd consider the improvement
                   negligible.
                   Example: For segmentation, I do not care for improvements less than 0.05% IoU in general.
                            But in kaggle competitions, even 0.01% would matter.

        patience: Patience is the number of val epochs to wait for to see an improvement. It is affected by the
                  ``check_val_every_n_epoch`` and ``val_check_interval`` params to the PL Trainer.

                  Takes experimentation to figure out appropriate patience for your project. Train the model
                  once without early stopping and see how long it takes to converge on a given dataset.
                  Choose the number of epochs between when you feel it's started to converge and after you're
                  sure the model has converged. Reduce the patience if you see the model continues to train for too long.

        verbose: Minimal extra info logs about earlystopping starting

        mode: Choose between "max" and "min". If the performance is considered better when metric is higher, choose
              "max", else "min".

        strict: Whether to crash the training if monitor is not found in the validation metrics. This should always be
                True. If early stopping is not desired, disable it.
    """

    def __init__(
        self,
        monitor="Val/mIoU",
        min_delta=0.0005,
        patience=10,
        verbose=True,
        mode="max",
        strict=True,
    ):
        super(EarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
        )
        self.verbose = verbose

    def setup(self, trainer, pl_module, stage: str):
        if self.verbose:
            pl_module.print(f"Initializing Callback {EarlyStopping.__name__}")


class LogMedia(Callback):
    """Logs model output images and other media to weights and biases

    Args:
        logging_epoch_interval (int): If > 0, log every N epochs. It will extract samples from the first batch.
        logging_batch_interval (int): If > 0, log every N batches (i.e. steps)
        max_images_to_log (int): Max number of images to extract from a batch to log.
        save_to_disk (boolr): If True, save results to disk.
        logs_dir (str or Path): Path to directory where results will be saved.
    """

    def __init__(
        self,
        max_images_to_log: int = 10,
        logging_epoch_interval: int = 1,
        logging_batch_interval: int = 0,
        save_to_disk: bool = True,
        logs_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.max_images_to_log = max_images_to_log
        self.logging_epoch_interval = logging_epoch_interval
        self.logging_batch_interval = logging_batch_interval
        self.save_to_disk = save_to_disk
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.flag_warn_once = False

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

        if self.verbose:
            pl_module.print(f"Initializing Callback {LogMedia.__name__}")

        if trainer.is_global_zero:
            if self.save_to_disk:
                if self.logs_dir is None:
                    raise ValueError(
                        f"Callback {LogMedia.__name__}: Invalid logs_dir: {self.logs_dir}. Please give "
                        f"valid path for logs_dir"
                    )
                else:
                    self.logs_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Save media
        if self._should_log(pl_module, batch_idx):
            self._log_images_to_wandb(trainer, pl_module, Mode.TRAIN)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Save media
        if self._should_log(pl_module, batch_idx):
            self._log_images_to_wandb(trainer, pl_module, Mode.VAL)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Save media
        if self._should_log(pl_module, batch_idx):
            self._log_images_to_wandb(trainer, pl_module, Mode.TEST)

    def on_test_epoch_end(self, trainer, pl_module):
        # Save final results to disk
        if self.save_to_disk:
            self._save_results_to_disk(pl_module, Mode.TEST)

    def _get_preds_from_lightningmodule(self, pl_module, mode: Mode):
        # Get all the latest batches from the data queue in LightningModule
        media_data = []
        log_media = pl_module.log_media[mode]
        while len(log_media) > 0:
            media_data.append(log_media.popleft())
        if len(media_data) == 0:
            pl_module.print("WARN: LogMedia Callback: log_media queue empty, no samples to log")
            return None

        inputs = torch.cat([x["inputs"] for x in media_data], dim=0)
        labels = torch.cat([x["labels"] for x in media_data], dim=0)
        preds = torch.cat([x["preds"] for x in media_data], dim=0)

        # Limit the num of samples and convert to numpy
        inputs = inputs[: self.max_images_to_log].detach().cpu().numpy().transpose((0, 2, 3, 1))
        inputs = (inputs * 255).astype(np.uint8)
        labels = labels[: self.max_images_to_log].detach().cpu().numpy().astype(np.uint8)
        preds = preds[: self.max_images_to_log].detach().cpu().numpy().astype(np.uint8)

        return inputs, labels, preds

    @rank_zero_only
    def _save_results_to_disk(self, pl_module, mode: Mode):
        """For a given mode (train/val/test), save the results to disk"""
        # Get the latest batches from the data queue in LightningModule
        data_r = self._get_preds_from_lightningmodule(pl_module, mode)
        if data_r is None:
            return
        else:
            inputs, labels, preds = data_r

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
        fname = str(self.logs_dir / f"results.{mode.name.lower()}.png")
        pl_module.print(f"Savings results to disk: {fname}")
        cv2.imwrite(fname, cv2.cvtColor(grid_results, cv2.COLOR_RGB2BGR))

    @rank_zero_only
    def _log_images_to_wandb(self, trainer, pl_module, mode: Mode = Mode.TRAIN):
        """Log images to wandb at the end of a batch. Steps are common for train/val/test"""
        if not self._logger_is_wandb(trainer):
            return

        # Get the latest batches from the data queue in LightningModule
        data_r = self._get_preds_from_lightningmodule(pl_module, mode)
        if data_r is None:
            return
        else:
            inputs, labels, preds = data_r

        # Create wandb Image for logging
        mask_list = []
        for img, lbl, pred in zip(inputs, labels, preds):
            mask_img = wandb.Image(
                img,
                masks={
                    "predictions": {"mask_data": pred, "class_labels_lapa": self.class_labels_lapa},
                    "groud_truth": {"mask_data": lbl, "class_labels_lapa": self.class_labels_lapa},
                },
            )
            mask_list.append(mask_img)

        wandb_log_label = f"{mode.value}/Predictions"
        trainer.logger.experiment.log({wandb_log_label: mask_list}, commit=False)

    def _should_log(self, pl_module, batch_idx: int) -> bool:
        """Returns True if logging should occur at this step and device.
        Logging occurs only on Global Rank 0 every N steps/epochs"""
        should_continue = False
        epoch_idx = pl_module.current_epoch
        if self.logging_epoch_interval > 0 and epoch_idx > 0:
            # Only log once per epoch, on 1st batch
            if ((epoch_idx + 1) % self.logging_epoch_interval == 0) and (batch_idx == 0):
                should_continue = True

        if self.logging_batch_interval > 0 and batch_idx > 0:
            if (batch_idx + 1) % self.logging_batch_interval == 0:
                should_continue = True

        return should_continue

    def _logger_is_wandb(self, trainer):
        """This callback only works with wandb logger.
        Skip if any other logger detected with warning"""
        if isinstance(trainer.logger, pl_loggers.base.DummyLogger) or trainer.running_sanity_check:
            # DummyLogger is used on processes other than rank0. Ignore it.
            return False

        if not isinstance(trainer.logger, pl_loggers.WandbLogger):
            if not self.flag_warn_once:
                # Give warning print only once to prevent clutter.
                print(
                    f"WARN: LogMedia only works with wandb logger. Current logger: {trainer.logger}. "
                    f"Will not log any media to wandb this run"
                )
                self.flag_warn_once = True
            return False

        return True

    @classmethod
    def get_log_media_structure(cls, log_media_max_batches: int) -> Dict[Mode, deque]:
        """Create a data structure for LogMedia"""
        log_media = {
            Mode.TRAIN: deque(maxlen=log_media_max_batches),
            Mode.VAL: deque(maxlen=log_media_max_batches),
            Mode.TEST: deque(maxlen=log_media_max_batches),
        }
        return log_media
