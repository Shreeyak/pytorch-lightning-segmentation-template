from abc import ABC
from typing import Optional

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import validate_config_group_generic


@dataclass(frozen=True)
class LoadWeightsConf(ABC):
    name: str
    path: Optional[str] = None


valid_names = {
    "disabled": LoadWeightsConf,
    "load_weights": LoadWeightsConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> LoadWeightsConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="load_weights"
    )
    return validated_dataclass
