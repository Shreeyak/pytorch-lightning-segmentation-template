import torch
import pytorch_lightning as pl
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pathlib import Path

from seg_lapa.networks.deeplab.deeplab import DeepLab
from seg_lapa.loss_func import CrossEntropy2D
from seg_lapa.datasets.lapa import LaPaDataModule
from seg_lapa.config_parse.train_conf import TrainConfig

class DeeplabV3plus(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = DeepLab(backbone='drn', output_stride=8, num_classes=11,
                             sync_bn=False, enable_amp=False)
        self.cross_entropy_loss = CrossEntropy2D(loss_per_image=True, ignore_index=255)

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

        # to aggregate epoch metrics use self.log or a metric. self.log logs metrics for each training_step.
        # It also logs the average across the epoch, to the progress bar and logger
        # "train_loss" is a reserved keyword
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@hydra.main(config_path='config', config_name='train')
def main(cfg: TrainConfig):
    print(OmegaConf.to_yaml(OmegaConf.to_container(cfg)))
    print(cfg)

    model = DeeplabV3plus()

    # Dataloaders
    data_conf = instantiate(cfg.dataset)
    dm = data_conf.get_datamodule()
    print(dm)
    exit()

    trainer = pl.Trainer(gpus=[0], overfit_batches=0.0,
                         distributed_backend="ddp", num_nodes=1,
                         precision=32,
                         limit_train_batches=1.0,
                         limit_val_batches=1.0,
                         limit_test_batches=1.0,
                         max_steps=cfg.num_steps,
                         fast_dev_run=False,
                         )
    trainer.fit(model, datamodule=dm)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    main()
