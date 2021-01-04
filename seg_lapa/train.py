from typing import Any, List

import hydra
import numpy as np
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers

from seg_lapa import metrics
from seg_lapa.callbacks.log_media import LogMediaQueue, Mode
from seg_lapa.config_parse.train_conf import ParseConfig
from seg_lapa.loss_func import CrossEntropy2D
from seg_lapa.utils import utils
from seg_lapa.utils.utils import is_rank_zero


class DeeplabV3plus(pl.LightningModule, ParseConfig):
    def __init__(self, cfg: DictConfig, log_media_max_batches=1):
        super().__init__()
        self.save_hyperparameters()  # Will save the config to wandb too
        # Accessing cfg via hparams allows value to be loaded from checkpoints
        self.config = self.parse_config(self.hparams.cfg)

        self.cross_entropy_loss = CrossEntropy2D(loss_per_image=True, ignore_index=255)
        self.model = self.config.model.get_model()

        self.iou_train = metrics.Iou(num_classes=self.config.model.num_classes)
        self.iou_val = metrics.Iou(num_classes=self.config.model.num_classes)
        self.iou_test = metrics.Iou(num_classes=self.config.model.num_classes)

        # Logging media such a images using `self.log()` is extremely memory-expensive.
        # Save predictions to be logged within a circular queue, to be consumed in the LogMedia callback.
        self.log_media: LogMediaQueue = LogMediaQueue(log_media_max_batches)

    def forward(self, x):
        """In lightning, forward defines the prediction/inference actions.
        This method can be called elsewhere in the LightningModule with: `outputs = self(inputs)`.
        """
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        """Defines the train loop. It is independent of forward().
        Don’t use any cuda or .to(device) calls in the code. PL will move the tensors to the correct device.
        """
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)

        """Log the value on GPU0 per step. Also log average of all steps at epoch_end."""
        self.log("Train/loss", loss, on_step=True, on_epoch=True)
        """Log the avg. value across all GPUs per step. Also log average of all steps at epoch_end.
        Alternately, you can use the ops 'sum' or 'avg'.
        Using sync_dist is efficient. It adds extremely minor overhead for scalar values.
        """
        # self.log("Train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        self.iou_train(predictions, labels)

        # Returning images is expensive - All the batches are accumulated for _epoch_end().
        # Save the latst predictions to be logged in an attr. They will be consumed by the LogMedia callback.
        self.log_media.append({"inputs": inputs, "labels": labels, "preds": predictions}, Mode.TRAIN)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("Val/loss", loss)

        # Calculate Metrics
        self.iou_val(predictions, labels)

        # Save the latest predictions to be logged
        self.log_media.append({"inputs": inputs, "labels": labels, "preds": predictions}, Mode.VAL)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("Test/loss", loss)

        # Calculate Metrics
        self.iou_test(predictions, labels)

        # Save the latest predictions to be logged
        self.log_media.append({"inputs": inputs, "labels": labels, "preds": predictions}, Mode.TEST)

        return {"test_loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # Compute and log metrics across epoch
        metrics_avg = self.iou_train.compute()
        self.log("Train/mIoU", metrics_avg.miou)
        self.iou_train.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        # Compute and log metrics across epoch
        metrics_avg = self.iou_val.compute()
        self.log("Val/mIoU", metrics_avg.miou)
        self.iou_val.reset()

    def test_epoch_end(self, outputs: List[Any]):
        # Compute and log metrics across epoch
        metrics_avg = self.iou_test.compute()
        self.log("Test/mIoU", metrics_avg.miou)
        self.log("Test/Accuracy", metrics_avg.accuracy.mean())
        self.log("Test/Precision", metrics_avg.precision.mean())
        self.log("Test/Recall", metrics_avg.recall.mean())

        # Save test results as a Table (WandB)
        self.log_results_table_wandb(metrics_avg)
        self.iou_test.reset()

    def log_results_table_wandb(self, metrics_avg: metrics.IouMetric):
        if not isinstance(self.logger, pl_loggers.WandbLogger):
            return

        results = metrics.IouMetric(
            iou_per_class=metrics_avg.iou_per_class.cpu().numpy(),
            miou=metrics_avg.miou.cpu().numpy(),
            accuracy=metrics_avg.accuracy.cpu().numpy().mean(),
            precision=metrics_avg.precision.cpu().numpy().mean(),
            recall=metrics_avg.recall.cpu().numpy().mean(),
            specificity=metrics_avg.specificity.cpu().numpy().mean(),
        )

        data = np.stack(
            [results.miou, results.accuracy, results.precision, results.recall, results.specificity], axis=0
        )
        data_l = [round(x.item(), 4) for x in data]
        table = wandb.Table(data=data_l, columns=["mIoU", "Accuracy", "Precision", "Recall", "Specificity"])
        self.logger.experiment.log({f"Test/Results": table}, commit=False)

        data = np.stack((np.arange(results.iou_per_class.shape[0]), results.iou_per_class)).T
        table = wandb.Table(data=data.round(decimals=4).tolist(), columns=["Class ID", "IoU"])
        self.logger.experiment.log({f"Test/IoU_per_class": table}, commit=False)

    def configure_optimizers(self):
        optimizer = self.config.optimizer.get_optimizer(self.parameters())

        ret_opt = {"optimizer": optimizer}

        sch = self.config.scheduler.get_scheduler(optimizer)
        if sch is not None:
            scheduler = {
                "scheduler": sch,  # The LR scheduler instance (required)
                "interval": "epoch",  # The unit of the scheduler's step size
                "frequency": 1,  # The frequency of the scheduler
                "reduce_on_plateau": False,  # For ReduceLROnPlateau scheduler
                "monitor": "Val/mIoU",  # Metric for ReduceLROnPlateau to monitor
                "strict": True,  # Whether to crash the training if `monitor` is not found
                "name": None,  # Custom name for LearningRateMonitor to use
            }

            ret_opt.update({"lr_scheduler": scheduler})

        return ret_opt


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    # if is_rank_zero():
    #     print("\nGiven Config:\n", OmegaConf.to_yaml(cfg))

    config = ParseConfig.parse_config(cfg)
    if is_rank_zero():
        print("\nResolved Dataclass:\n", config, "\n")

    utils.fix_seeds(config.random_seed)
    exp_dir = utils.generate_log_dir_path(config)

    wb_logger = config.logger.get_logger(cfg, config.logs_root_dir)
    callbacks = config.callbacks.get_callbacks_list(exp_dir, cfg)
    dm = config.dataset.get_datamodule()

    # Load weights
    if config.load_weights.path is None:
        model = DeeplabV3plus(cfg)
    else:
        model = DeeplabV3plus.load_from_checkpoint(config.load_weights.path, cfg=cfg)

    trainer = config.trainer.get_trainer(wb_logger, callbacks, config.logs_root_dir)

    # Run Training
    trainer.fit(model, datamodule=dm)

    # Run Testing
    result = trainer.test(ckpt_path=None)  # Prints the final result

    wandb.finish()


if __name__ == "__main__":
    main()
