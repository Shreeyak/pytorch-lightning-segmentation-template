from abc import ABC, abstractmethod
from typing import Optional, Union, List

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from torch.optim.optimizer import Optimizer


@dataclass
class SchedulerConf(ABC):
    name: str

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        pass


@dataclass()
class DisabledConfig(SchedulerConf):
    def get_scheduler(self, optimizer: Optimizer) -> None:
        return None


@dataclass()
class CyclicConfig(SchedulerConf):
    base_lr: float
    max_lr: float
    step_size_up: int
    step_size_down: Optional[int]

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.CyclicLR:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                 cycle_momentum=False,
                                                 **args)


@dataclass()
class PolyConfig(SchedulerConf):
    max_iter: int
    pow_factor: float

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        max_iter = float(self.max_iter)
        pow_factor = float(self.pow_factor)

        def poly_schedule(n_iter: int) -> float:
            return (1 - n_iter / max_iter) ** pow_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schedule)


@dataclass()
class StepConfig(SchedulerConf):
    step_size: int
    gamma: float

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.StepLR:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return torch.optim.lr_scheduler.StepLR(optimizer, **args)


@dataclass()
class PlateauConfig(SchedulerConf):
    factor: float
    patience: int
    min_lr: Union[float, List[float]]
    mode: str
    threshold: float
    cooldown: int
    eps: float
    verbose: bool

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        # Clean the arguments
        args = vars(self)
        args.pop('name', None)
        args.pop('__initialised__', None)

        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **args)


valid_options = {
    "disabled": DisabledConfig,
    "cyclic": CyclicConfig,
    "plateau": PlateauConfig,
    "poly": PolyConfig,
    "step": StepConfig,
}


def validate_schedulerconf(cfg_scheduler: DictConfig) -> SchedulerConf:
    try:
        schedulerconf = valid_options[cfg_scheduler.name](**cfg_scheduler)
    except KeyError:
        raise ValueError(f"Invalid Config: '{cfg_scheduler.name}' is not a valid scheduler. "
                         f"Valid Options: {list(valid_options.keys())}")

    return schedulerconf
