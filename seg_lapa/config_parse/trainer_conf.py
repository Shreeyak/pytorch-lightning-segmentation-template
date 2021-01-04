from abc import ABC, abstractmethod
from typing import List, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass(frozen=True)
class TrainerConf(ABC):
    name: str

    @abstractmethod
    def get_trainer(
        self, pl_logger: LightningLoggerBase, callbacks: List[Callback], default_root_dir: str
    ) -> pl.Trainer:
        pass


@dataclass(frozen=True)
class TrainerConfig(TrainerConf):
    gpus: int
    accelerator: Optional[str]
    precision: int
    max_epochs: int
    resume_from_checkpoint: Optional[str]
    log_every_n_steps: int

    benchmark: bool = False
    deterministic: bool = False
    fast_dev_run: bool = False
    overfit_batches: float = 0.0
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0

    def get_trainer(
        self, pl_logger: LightningLoggerBase, callbacks: List[Callback], default_root_dir: str
    ) -> pl.Trainer:
        trainer = pl.Trainer(
            logger=pl_logger,
            callbacks=callbacks,
            default_root_dir=default_root_dir,
            **asdict_filtered(self),
        )
        return trainer


valid_names = {"trainer": TrainerConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> TrainerConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="trainer"
    )
    return validated_dataclass
