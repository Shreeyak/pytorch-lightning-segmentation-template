import albumentations as A
import torch
import pytorch_lightning as pl
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from .networks.deeplab.deeplab import DeepLab
from .loss_func import CrossEntropy2D
from .dataloaders import LapaDataset


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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig):
    model = DeeplabV3plus()

    # Dataloaders
    resize_h = 256
    resize_w = 256
    augs_test = A.Compose([
        # Geometric Augs
        A.SmallestMaxSize(max_size=resize_h, interpolation=0, p=1.0),
        A.CenterCrop(height=resize_h, width=resize_w, p=1.0),
    ])

    lapa_train = LapaDataset(image_dir=cfg.dataset.lapa.train.image_dir, label_dir=cfg.dataset.lapa.train.label_dir,
                             augmentations=augs_test)
    lapa_test = LapaDataset(image_dir=cfg.dataset.lapa.test.image_dir, label_dir=cfg.dataset.lapa.test.label_dir,
                            augmentations=augs_test)
    lapa_val = LapaDataset(image_dir=cfg.dataset.lapa.val.image_dir, label_dir=cfg.dataset.lapa.val.label_dir,
                           augmentations=augs_test)

    train_loader = DataLoader(lapa_train, batch_size=cfg.batch_size.train, num_workers=4)
    val_loader = DataLoader(lapa_test, batch_size=cfg.batch_size.test, num_workers=4)
    test_loader = DataLoader(lapa_val, batch_size=cfg.batch_size.test, num_workers=4)

    trainer = pl.Trainer(gpus=[1], overfit_batches=0.0,
                         limit_train_batches=cfg.dataset.lapa.train.use_factor,
                         limit_val_batches=cfg.dataset.lapa.val.use_factor,
                         limit_test_batches=cfg.dataset.lapa.test.use_factor,
                         max_steps=cfg.num_steps,
                         fast_dev_run=False,
                         )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    main()
