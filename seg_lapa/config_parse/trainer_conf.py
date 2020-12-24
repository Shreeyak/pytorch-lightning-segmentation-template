from abc import ABC, abstractmethod
from typing import Optional, Union, List

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import cleaned_asdict, validate_config_group_generic


@dataclass(frozen=True)
class TrainerConf(ABC):
    name: str

    @abstractmethod
    def get_trainer(self) -> pl.Trainer:
        pass


@dataclass(frozen=True)
class TrainerConfig(TrainerConf):
    gpus: Union[int, str, List[int]]
    overfit_batches: Union[int, float]
    distributed_backend: Optional[str]
    precision: int
    limit_train_batches: Union[int, float]
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    fast_dev_run: Union[int, bool]
    max_epochs: int
    resume_from_checkpoint: Optional[str]

    def get_trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(**cleaned_asdict(self))
        return trainer


valid_names = {"trainer": TrainerConfig}


def validate_config_group(cfg_subgroup: DictConfig) -> TrainerConf:
    validated_dataclass = validate_config_group_generic(cfg_subgroup,
                                                        mapping_names_dataclass=valid_names,
                                                        config_category='trainer')
    return validated_dataclass
