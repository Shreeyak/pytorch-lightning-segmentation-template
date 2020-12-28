from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning.callbacks import ModelCheckpoint

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass(frozen=True)
class CheckpointConf(ABC):
    name: str

    @abstractmethod
    def get_checkpoint_callback(self) -> pl.callbacks:
        pass


@dataclass(frozen=True)
class CheckpointConfig(CheckpointConf):
    dir_path: Optional[str]
    filename: Optional[str]

    def get_checkpoint_callback(self) -> pl.callbacks:
        checkpoint_callback = ModelCheckpoint(**asdict_filtered(self))
        return checkpoint_callback


valid_names = {"checkpoint_callback": CheckpointConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> CheckpointConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="checkpoint_callback"
    )
    return validated_dataclass
