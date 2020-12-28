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
    def get_trainer(self, pl_logger: LightningLoggerBase, checkpoint_callback: pl.callbacks) -> pl.Trainer:
        pass


@dataclass(frozen=True)
class TrainerConfig(TrainerConf):
    gpus: int
    overfit_batches: Union[float, int]
    distributed_backend: Optional[str]
    precision: int
    limit_train_batches: Union[float, int]
    limit_val_batches: Union[float, int]
    limit_test_batches: Union[float, int]
    fast_dev_run: Union[int, bool]
    max_epochs: int
    resume_from_checkpoint: Optional[str]
    log_every_n_steps: int

    def get_trainer(self, pl_logger, checkpoint_callback) -> pl.Trainer:
        trainer = pl.Trainer(logger=pl_logger, callbacks=[checkpoint_callback], **asdict_filtered(self))
        return trainer


valid_names = {"trainer": TrainerConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> TrainerConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="trainer"
    )
    return validated_dataclass
