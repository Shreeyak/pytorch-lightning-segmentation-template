import os

import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import torch

from seg_lapa.loss_func import CrossEntropy2D
from seg_lapa.config_parse.train_conf import TrainConf
from seg_lapa.config_parse import train_conf
from seg_lapa import metrics


class DeeplabV3plus(pl.LightningModule):
    def __init__(self, config: TrainConf):
        super().__init__()
        self.cross_entropy_loss = CrossEntropy2D(loss_per_image=True, ignore_index=255)
        self.config = config
        self.model = self.config.model.get_model()
        self.iou_meter = metrics.IouMetric(num_classes=config.model.num_classes)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # don’t use any cuda or .to(device) calls in code
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        batch_loss = loss / len(batch[0])
        wandb.log({"Train/BatchWise Loss": batch_loss})

        # to aggregate epoch metrics use self.log or a metric. self.log logs metrics for each training_step.
        # It also logs the average across the epoch, to the progress bar and logger
        # "train_loss" is a reserved keyword
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        batch_loss = loss / len(batch[0])

        wandb.log({"Val/BatchWise Loss": batch_loss})

        return {
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        batch_loss = loss / len(batch[0])
        wandb.log({"Test/Loss": batch_loss})

        return {
            "test_loss": loss,
        }

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        wandb.log({"Train/Epoch Loss": loss})

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        wandb.log({"Val/Epoch Loss": loss})

    def configure_optimizers(self):
        optimizer = self.config.optimizer.get_optimizer(self.parameters())
        return optimizer


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\nGiven Config:\n", OmegaConf.to_yaml(cfg))

    config = train_conf.parse_config(cfg)

    logger_wandb = config.logger.get_logger(cfg)

    model = DeeplabV3plus(config)

    trainer = config.trainer.get_trainer()
    dm = config.dataset.get_datamodule()
    trainer.fit(model, datamodule=dm)
    result = trainer.test()  # Prints the final result


if __name__ == "__main__":
    main()
