from abc import ABC, abstractmethod

import pytorch_lightning as pl
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass


@dataclass
class TrainerConf(ABC):
    name: str

    @abstractmethod
    def get_trainer(self) -> pl.Trainer:
        pass


@dataclass
class TrainerConfig(TrainerConf):
    gpus: str
    overfit_batches: float
    distributed_backend: str
    num_nodes: int
    precision: int
    limit_train_batches: float
    limit_val_batches: float
    limit_test_batches: float
    fast_dev_run: bool

    def get_trainer(self) -> pl.Trainer:
        # Clean the arguments
        args = vars(self)
        args.pop('name')
        args.pop("__initialised__")

        trainer = pl.Trainer(**args)

        return trainer


valid_options = {"trainer": TrainerConfig}


def validate_trainerconf(cfg_trainer: DictConfig) -> TrainerConf:
    try:
        trainerconf = valid_options[cfg_trainer.name](**cfg_trainer)
    except KeyError:
        raise ValueError(
            f"Invalid Config for trainer. "
            f"Valid Options: {list(valid_options.keys())}"
        )

    return trainerconf
