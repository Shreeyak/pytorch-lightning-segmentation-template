from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic


@dataclass(frozen=True)
class OptimConf(ABC):
    name: str

    @abstractmethod
    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        pass


@dataclass(frozen=True)
class AdamConf(OptimConf):
    lr: float
    weight_decay: float

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=model_params, **asdict_filtered(self))


@dataclass(frozen=True)
class SgdConf(OptimConf):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=model_params, **asdict_filtered(self))


valid_names = {"adam": AdamConf, "sgd": SgdConf}


def validate_config_group(cfg_subgroup: DictConfig) -> OptimConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="optimizer"
    )
    return validated_dataclass
