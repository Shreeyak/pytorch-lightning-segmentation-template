import pytorch_lightning as pl
import numpy as np
import wandb
from pytorch_lightning.callbacks import early_stopping, Callback

from seg_lapa.utils.segmentation_label2rgb import LabelToRGB, Palette


class EarlyStopping(early_stopping.EarlyStopping):
    """Direct sub-class of PL's early-stopping, changing the default parameters to suit our project

    Args:
        monitor: Monitor a key validation metric (IoU). Monitoring loss is not a good idea as it is an unreliable
                 indicator of model performance. Two models might have the same loss but different
                 performance (IoU), or the loss might start increasing, even though IoU does not decrease.

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


class LogMedia(Callback):
    def __init__(self, logging_batch_interval: int = 1, max_images_to_log: int = 10):
        super().__init__()
        self.logging_batch_interval = logging_batch_interval
        self.max_images_to_log = max_images_to_log
        self.label2rgb = LabelToRGB()
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
        self.flag_warn_once = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._common_batch_end(trainer, outputs, batch, "Train/Predictions")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._common_batch_end(trainer, outputs, batch, "Val/Predictions")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._common_batch_end(trainer, outputs, batch, "Test/Predictions")

    def _common_batch_end(self, trainer, outputs, batch, wandb_log_label="Train/Predictions"):
        """Log images to wandb at the end of a batch. Steps are common for train/val/test"""
        # Log images only every N batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:
            return

        self._check_for_wandb_logger(trainer)

        mask_list = self._get_wandb_image_from_batch(batch, outputs)
        trainer.logger.experiment.log({wandb_log_label: mask_list}, commit=False)

    def _check_for_wandb_logger(self, trainer):
        """This callback only works with wandb logger.
        Skip if any other logger detected with warning"""
        if trainer.logger is None:
            return None
        if isinstance(trainer.logger.experiment, pl.loggers.base.DummyExperiment):
            return None
        if not isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            if not self.flag_warn_once:
                # Given warning print only once. To prevent clutter.
                print(
                    f"WARN: LogMedia only works with wandb logger. Current logger: {trainer.logger.experiment}. "
                    f"Will not log any media this run"
                )
                self.flag_warn_once = True
            return None

    def _get_wandb_image_from_batch(self, batch, outputs):
        # Pick the batch from GPU0, Dataloader0
        inputs, labels = batch
        predictions = outputs[0][0]["extra"]["preds"]

        # Limit the num of samples and convert to numpy
        inputs = inputs[: self.max_images_to_log].detach().cpu().numpy().transpose((0, 2, 3, 1))
        labels = labels[: self.max_images_to_log].detach().cpu().numpy().astype(np.uint8)
        predictions = predictions[: self.max_images_to_log].detach().cpu().numpy().astype(np.uint8)

        # Create wandb Image for logging
        mask_list = []
        for img, lbl, pred in zip(inputs, labels, predictions):
            mask_img = wandb.Image(
                img,
                masks={
                    "predictions": {"mask_data": pred, "class_labels_lapa": self.class_labels_lapa},
                    "groud_truth": {"mask_data": lbl, "class_labels_lapa": self.class_labels_lapa},
                },
            )
            mask_list.append(mask_img)

        return mask_list
