from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from seg_lapa.datasets.lapa import LaPaDataModule


@dataclass(frozen=True)
class DatasetConf(ABC):
    name: str

    @abstractmethod
    def get_datamodule(self) -> pl.LightningDataModule:
        pass


@dataclass(frozen=True)
class LapaConf(DatasetConf):
    data_dir: str
    batch_size: int
    num_workers: int
    resize_h: int
    resize_w: int

    def get_datamodule(self) -> LaPaDataModule:
        return LaPaDataModule(**asdict_filtered(self))


valid_names = {"lapa": LapaConf}


def validate_config_group(cfg_subgroup: DictConfig) -> DatasetConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="dataset"
    )
    return validated_dataclass
