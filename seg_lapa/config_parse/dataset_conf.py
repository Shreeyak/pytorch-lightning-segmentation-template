from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from datasets.lapa import LaPaDataModule


@dataclass
class DatasetConf(ABC):
    name: str

    @abstractmethod
    def get_datamodule(self) -> pl.LightningDataModule:
        pass


@dataclass
class LapaConf(DatasetConf):
    data_dir: str
    batch_size: int
    num_workers: int
    resize_h: int
    resize_w: int

    def get_datamodule(self) -> LaPaDataModule:
        # Clean the arguments
        args = vars(self)
        args.pop('name')
        args.pop('__initialised__')

        return LaPaDataModule(**args)


valid_options = {
    "lapa": LapaConf
}


def validate_dataconf(cfg_dataset: DictConfig) -> DatasetConf:
    try:
        dataconf = valid_options[cfg_dataset.name](**cfg_dataset)
    except KeyError:
        raise ValueError(f"Invalid Config: '{cfg_dataset.name}' is not a valid dataset. "
                         f"Valid Options: {list(valid_options.keys())}")

    return dataconf
