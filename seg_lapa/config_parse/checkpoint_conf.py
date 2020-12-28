from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning.callbacks import ModelCheckpoint

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from seg_lapa.utils.path_check import get_project_root


@dataclass
class CheckpointConf(ABC):
    name: str

    @abstractmethod
    def get_checkpoint_callback(self, *args) -> pl.callbacks:
        pass


@dataclass
class CheckpointConfig(CheckpointConf):
    dirpath: Optional[str]

    def get_checkpoint_callback(self, cfg: DictConfig, run_id) -> pl.callbacks:
        if self.dirpath is None:
            self.dirpath = str(get_project_root())+'/logs/'+run_id

        checkpoint_callback = ModelCheckpoint(**asdict_filtered(self))
        return checkpoint_callback


valid_names = {"checkpoint_callback": CheckpointConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> CheckpointConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="checkpoint_callback"
    )
    return validated_dataclass
