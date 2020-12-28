from abc import ABC, abstractmethod
from typing import Optional, Union, List

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning.loggers.base import LightningLoggerBase

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass(frozen=True)
class TrainerConf(ABC):
    name: str

    @abstractmethod
    def get_trainer(self, pl_logger: LightningLoggerBase) -> pl.Trainer:
        pass


@dataclass(frozen=True)
class TrainerConfig(TrainerConf):
    gpus: int
    accelerator: Optional[str]
    precision: int
    fast_dev_run: Union[int, bool]
    max_epochs: int
    resume_from_checkpoint: Optional[str]
    log_every_n_steps: int
    overfit_batches: Union[int, float] = 0.0
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0

    def get_trainer(self, pl_logger) -> pl.Trainer:
        trainer = pl.Trainer(logger=pl_logger, **asdict_filtered(self))
        return trainer


valid_names = {"trainer": TrainerConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> TrainerConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="trainer"
    )
    return validated_dataclass
