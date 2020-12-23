from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass


@dataclass
class OptimConf(ABC):
    name: str

    @abstractmethod
    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        pass


@dataclass
class AdamConf(OptimConf):
    lr: float
    weight_decay: float

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return torch.optim.Adam(params=model_params, **args)


@dataclass
class SgdConf(OptimConf):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return torch.optim.SGD(params=model_params, **args)


valid_options = {
    "adam": AdamConf,
    "sgd": SgdConf
}


def validate_optimconf(cfg_optim: DictConfig) -> OptimConf:
    try:
        optimconf = valid_options[cfg_optim.name](**cfg_optim)
    except KeyError:
        raise ValueError(f"Invalid Config: '{cfg_optim.name}' is not a valid optimizer. "
                         f"Valid Options: {list(valid_options.keys())}")

    return optimconf
