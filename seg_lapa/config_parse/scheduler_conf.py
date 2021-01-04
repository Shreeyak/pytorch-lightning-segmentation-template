from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from torch.optim.optimizer import Optimizer

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass(frozen=True)
class SchedulerConf(ABC):
    name: str

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        pass


@dataclass(frozen=True)
class DisabledConfig(SchedulerConf):
    def get_scheduler(self, optimizer: Optimizer) -> None:
        return None


@dataclass(frozen=True)
class CyclicConfig(SchedulerConf):
    base_lr: float
    max_lr: float
    step_size_up: int
    step_size_down: Optional[int]

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.CyclicLR:
        return torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum=False, **asdict_filtered(self))


@dataclass(frozen=True)
class PolyConfig(SchedulerConf):
    max_iter: int
    pow_factor: float

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        max_iter = float(self.max_iter)
        pow_factor = float(self.pow_factor)

        def poly_schedule(n_iter: int) -> float:
            return (1 - n_iter / max_iter) ** pow_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schedule)


@dataclass(frozen=True)
class StepConfig(SchedulerConf):
    step_size: int
    gamma: float

    def get_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(optimizer, **asdict_filtered(self))


@dataclass(frozen=True)
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
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **asdict_filtered(self))


valid_names = {
    "disabled": DisabledConfig,
    "cyclic": CyclicConfig,
    "plateau": PlateauConfig,
    "poly": PolyConfig,
    "step": StepConfig,
}


def validate_config_group(cfg_subgroup: DictConfig) -> SchedulerConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="scheduler"
    )
    return validated_dataclass
