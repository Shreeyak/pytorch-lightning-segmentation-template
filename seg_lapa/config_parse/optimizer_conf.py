from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import cleaned_asdict, validate_config_group_generic


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
        return torch.optim.Adam(params=model_params, **cleaned_asdict(self))


@dataclass
class SgdConf(OptimConf):
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def get_optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.SGD(params=model_params, **cleaned_asdict(self))


valid_names = {
    "adam": AdamConf,
    "sgd": SgdConf
}


def validate_config_group(cfg_subgroup: DictConfig) -> OptimConf:
    validated_dataclass = validate_config_group_generic(cfg_subgroup,
                                                        mapping_names_dataclass=valid_names,
                                                        config_category='optimizer')
    return validated_dataclass
