from abc import ABC, abstractmethod
from typing import Dict

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from seg_lapa.callbacks import EarlyStopping


@dataclass(frozen=True)
class CallbacksConf(ABC):
    name: str

    @abstractmethod
    def get_callbacks_dict(self):
        pass


@dataclass(frozen=True)
class DisabledCallbacksConf(CallbacksConf):
    def get_callbacks_dict(self) -> Dict:
        return {}


@dataclass(frozen=True)
class EarlyStopConf:
    """Dataclass just to initialize and return the Early Stopping Callback"""

    min_delta: float
    patience: int

    def get_callback(self):
        args_dict = asdict_filtered(self)
        return EarlyStopping(**args_dict)


@dataclass(frozen=True)
class StandardCallbacksConf(CallbacksConf):
    """Get a dictionary of all the callbacks."""

    early_stopping: Dict

    def get_callbacks_dict(self) -> Dict:
        early_stop = EarlyStopConf(**self.early_stopping)
        return {"early_stopping": early_stop.get_callback()}


valid_names = {
    "disabled": DisabledCallbacksConf,
    "standard": StandardCallbacksConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> CallbacksConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="callback"
    )
    return validated_dataclass
