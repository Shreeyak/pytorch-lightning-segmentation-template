import os
from typing import List, Any

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf, DictConfig

from seg_lapa import metrics
from seg_lapa.config_parse import train_conf
from seg_lapa.config_parse.train_conf import TrainConf
from seg_lapa.loss_func import CrossEntropy2D


class DeeplabV3plus(pl.LightningModule):
    def __init__(self, config: TrainConf):
        super().__init__()
        self.cross_entropy_loss = CrossEntropy2D(loss_per_image=True, ignore_index=255)
        self.config = config
        self.model = self.config.model.get_model()
        # self.iou_meter = {
        #     "train": metrics.IouSync(num_classes=config.model.num_classes),
        #     "val": metrics.IouSync(num_classes=config.model.num_classes),
        #     "test": metrics.IouSync(num_classes=config.model.num_classes),
        # }
        self.iou_train = metrics.IouSync(num_classes=config.model.num_classes)
        self.iou_val = metrics.IouSync(num_classes=config.model.num_classes)
        self.iou_test = metrics.IouSync(num_classes=config.model.num_classes)

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

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        predictions = outputs.argmax(dim=1)

        # Calculate Loss
        loss = self.cross_entropy_loss(outputs, labels)
        self.log("Val/loss", loss, sync_dist=True, sync_dist_op="avg")

        # Calculate Metrics
        self.iou_val(predictions, labels)

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

        return {"test_loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        metrics_avg = self.iou_train.compute()
        self.log("Train/mIoU", metrics_avg.miou)

    def validation_epoch_end(self, outputs: List[Any]):
        metrics_avg = self.iou_val.compute()
        self.log("Val/mIoU", metrics_avg.miou)

    def test_epoch_end(self, outputs: List[Any]):
        metrics_avg = self.iou_test.compute()
        self.log("Test/mIoU", metrics_avg.miou)

    def configure_optimizers(self):
        optimizer = self.config.optimizer.get_optimizer(self.parameters())
        return optimizer


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nGiven Config:\n", OmegaConf.to_yaml(cfg))

    config = train_conf.parse_config(cfg)
    if local_rank == 0:
        print("\nResolved Dataclass:\n", config, "\n")

    wb_logger = config.logger.get_logger(cfg)
    trainer = config.trainer.get_trainer(wb_logger)
    model = DeeplabV3plus(config)
    dm = config.dataset.get_datamodule()

    # Run Training
    trainer.fit(model, datamodule=dm)

    # Run Testing
    result = trainer.test(ckpt_path=None)  # Prints the final result

    wandb.finish()


if __name__ == "__main__":
    main()
