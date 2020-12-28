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
        # Log images only every N batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:
            return

        # Only works with wandb logger
        if trainer.logger is None:
            return
        if not isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
            if not self.flag_warn_once:
                # Given warning print only once. To prevent clutter.
                print(
                    f"WARN: LogMedia only works with wandb logger. Current logger: {trainer.logger.experiment}. "
                    f"Will not log any media this run"
                )
            return

        # Pick the last batch and logits
        inputs, labels = batch
        print(f"inputs: {inputs.shape}, labels: {labels.shape}")
        predictions = outputs[0][0]["extra"]["preds"]

        # Limit the num of samples
        inputs = inputs[: self.max_images_to_log].cpu().numpy().transpose((0, 2, 3, 1))
        labels = labels[: self.max_images_to_log].cpu().numpy()
        predictions = predictions[: self.max_images_to_log].cpu().numpy()

        # Colorize labels and predictions
        # labels_rgb = self._colorize_batch_images(labels)
        # predictions_rgb = self._colorize_batch_images(predictions)

        # Log to wandb
        mask_list = []
        for img, lbl, pred in zip(inputs, labels, predictions):
            print(f"lbl: {lbl.shape}, preds: {pred.shape}")
            mask_img = wandb.Image(
                img,
                masks={
                    "predictions": {"mask_data": pred, "class_labels_lapa": self.class_labels_lapa},
                    "groud_truth": {"mask_data": lbl, "class_labels_lapa": self.class_labels_lapa},
                },
            )
            mask_list.append(mask_img)

        trainer.logger.experiment.log({"Train/Predictions": mask_list})

    def _colorize_batch_images(self, batch_label: np.ndarray):
        batch_label_rgb = [self.label2rgb.map_color_palette(label, Palette.LAPA) for label in batch_label]
        batch_label_rgb = np.stack(batch_label_rgb, axis=0)
        return batch_label_rgb
