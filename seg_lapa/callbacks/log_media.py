from collections import deque
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import Callback
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


class LogMedia(Callback):
    """Logs model output images and other media to weights and biases

    This callback required adding an attribute to the LightningModule called ``self.log_media``. This is a cicular
    queue that holds the latest N batches. This callback fetches the latest data from the queue for logging.

    Use ``get_empty_data_queue()`` to get the data structure.

    Usage:
        import pytorch_lightning as pl

        class MyModel(pl.LightningModule):
            self.log_media: Dict[Mode, deque] = LogMedia.get_empty_data_queue(log_media_max_batches)

        trainer = pl.Trainer(callbacks=[LogMedia()])

    Args:
        period_epoch (int): If > 0, log every N epochs
        period_step (int): If > 0, log every N steps (i.e. batches)
        max_samples (int): Max number of images to log
        save_to_disk (bool): If True, save results to disk
        logs_dir (str or Path): Path to directory where results will be saved
    """

    def __init__(
        self,
        max_samples: int = 10,
        period_epoch: int = 1,
        period_step: int = 0,
        save_to_disk: bool = True,
        logs_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.max_samples = max_samples
        self.period_epoch = period_epoch
        self.period_step = period_step
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

    # TODO: Replace this with a proper data-structure. Create class with methods to add/read each queue
    @classmethod
    def get_empty_data_queue(cls, log_media_max_batches: int) -> Dict[Mode, deque]:
        """Create a data structure for LogMedia"""
        log_media = {
            Mode.TRAIN: deque(maxlen=log_media_max_batches),
            Mode.VAL: deque(maxlen=log_media_max_batches),
            Mode.TEST: deque(maxlen=log_media_max_batches),
        }
        return log_media

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
                # else:
                #     self.logs_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.TRAIN)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.VAL)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self._should_log_step(trainer, batch_idx):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.TEST)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if not self._should_log_epoch(trainer):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.TRAIN)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.VAL)

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._should_log_epoch(trainer):
            return

        self._log_images_to_wandb(trainer, pl_module, Mode.TEST)
        self._save_results_to_disk(pl_module, Mode.TEST)

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

    def _logger_is_wandb(self, trainer):
        """This callback only works with wandb logger"""
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

    def _get_preds_from_lightningmodule(self, pl_module, mode: Mode):
        # Fetch latest N batches from the data queue in LightningModule
        media_data = []
        log_media = pl_module.log_media[mode]
        while len(log_media) > 0:
            media_data.append(log_media.popleft())
        if len(media_data) == 0:
            return None  # Queue empty

        inputs = torch.cat([x["inputs"] for x in media_data], dim=0)
        labels = torch.cat([x["labels"] for x in media_data], dim=0)
        preds = torch.cat([x["preds"] for x in media_data], dim=0)

        # Limit the num of samples and convert to numpy
        inputs = inputs[: self.max_samples].detach().cpu().numpy().transpose((0, 2, 3, 1))
        inputs = (inputs * 255).astype(np.uint8)
        labels = labels[: self.max_samples].detach().cpu().numpy().astype(np.uint8)
        preds = preds[: self.max_samples].detach().cpu().numpy().astype(np.uint8)

        return inputs, labels, preds

    @rank_zero_only
    def _save_results_to_disk(self, pl_module, mode: Mode):
        """For a given mode (train/val/test), save the results to disk"""
        if not self.save_to_disk:
            return

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

        wandb_log_label = f"{mode.name.title()}/Predictions"
        trainer.logger.experiment.log({wandb_log_label: mask_list}, commit=False)
