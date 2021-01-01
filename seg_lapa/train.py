from typing import Any, List

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig

from seg_lapa import metrics
from seg_lapa.config_parse import train_conf
from seg_lapa.config_parse.train_conf import TrainConf
from seg_lapa.loss_func import CrossEntropy2D
from seg_lapa.utils.path_check import get_project_root
from seg_lapa.callbacks.log_media import Mode, LogMediaQueue
from seg_lapa.utils.utils import is_rank_zero
from seg_lapa.utils import utils


class DeeplabV3plus(pl.LightningModule):
    def __init__(self, config: TrainConf, log_media_max_batches=1):
        super().__init__()
        self.cross_entropy_loss = CrossEntropy2D(loss_per_image=True, ignore_index=255)
        self.config = config
        self.model = self.config.model.get_model()

        self.iou_train = metrics.Iou(num_classes=config.model.num_classes)
        self.iou_val = metrics.Iou(num_classes=config.model.num_classes)
        self.iou_test = metrics.Iou(num_classes=config.model.num_classes)

        # Returning images from _step methods is memory-expensive. Save predictions to be logged in a circular queue
        # to be consumed in a callback.
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
        # self.log("Train/loss", loss, on_step=True, on_epoch=True)
        """Log the avg. value across all GPUs per step. Also log average of all steps at epoch_end.
        Alternately, you can use the ops 'sum' or 'avg'.
        Using sync_dist is efficient. It adds extremely minor overhead for scalar values.
        """
        self.log("Train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        self.iou_train(predictions, labels)

        # Returning images is expensive - All the batches are accumulated for _epoch_end().
        # Save the latst predictions to be logged in an attr. They will be consumed by the LogMedia callback.
        self.log_media.append(
            {
                "inputs": inputs.clone().detach(),
                "labels": labels.clone().detach(),
                "preds": predictions.clone().detach(),
            },
            Mode.TRAIN,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("Val/loss", loss, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        self.iou_val(predictions, labels)

        # Save the latest predictions to be logged
        self.log_media.append(
            {
                "inputs": inputs.clone().detach(),
                "labels": labels.clone().detach(),
                "preds": predictions.clone().detach(),
            },
            Mode.VAL,
        )

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("Test/loss", loss, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        self.iou_test(predictions, labels)

        # Save the latest predictions to be logged
        self.log_media.append(
            {
                "inputs": inputs.clone().detach(),
                "labels": labels.clone().detach(),
                "preds": predictions.clone().detach(),
            },
            Mode.TEST,
        )

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
        self.iou_test.reset()

    def configure_optimizers(self):
        optimizer = self.config.optimizer.get_optimizer(self.parameters())
        return optimizer


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    # if is_rank_zero():
    #     print("\nGiven Config:\n", OmegaConf.to_yaml(cfg))

    config = train_conf.parse_config(cfg)
    if is_rank_zero():
        print("\nResolved Dataclass:\n", config, "\n")

    utils.fix_seeds(config.random_seed)
    run_id = utils.generate_run_id(cfg)
    exp_dir = utils.create_log_dir(run_id, config.logs_root_dir)

    wb_logger = config.logger.get_logger(cfg, run_id, config.logs_root_dir)
    callbacks = config.callbacks.get_callbacks_list(exp_dir, cfg)
    trainer = config.trainer.get_trainer(wb_logger, callbacks, config.logs_root_dir)
    model = DeeplabV3plus(config)
    dm = config.dataset.get_datamodule()

    # Run Training
    trainer.fit(model, datamodule=dm)

    # Run Testing
    result = trainer.test(ckpt_path=None)  # Prints the final result

    wandb.finish()


if __name__ == "__main__":
    main()
