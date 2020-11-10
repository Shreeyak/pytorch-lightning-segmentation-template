from dataclasses import dataclass

import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from seg_lapa.datasets.lapa import LaPaDataModule


@dataclass
class DatasetConf:
    def get_datamodule(self) -> pl.LightningDataModule:
        pass


@dataclass
class LapaConf(DatasetConf):
    data_dir: str = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    resize_h: int = MISSING
    resize_w: int = MISSING
    _target_: str = MISSING

    def get_datamodule(self) -> LaPaDataModule:
        return LaPaDataModule(**vars(self))


cs = ConfigStore.instance()
cs.store(group="dataset/lapa", name="lapa", node=LapaConf)
