from dataclasses import dataclass

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class OptimConf:
    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        pass


@dataclass
class AdamConf(OptimConf):
    lr: float = MISSING
    weight_decay: float = MISSING

    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(parameters, **vars(self))


@dataclass
class SgdConf(OptimConf):
    lr: float = MISSING
    momentum: float = MISSING
    weight_decay: float = MISSING
    nesterov: bool = MISSING

    def get_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, **vars(self))


cs = ConfigStore.instance()
cs.store(group="optimizer/adam", name="adam", node=AdamConf)
cs.store(group="optimizer/sgd", name="sgd", node=SgdConf)
