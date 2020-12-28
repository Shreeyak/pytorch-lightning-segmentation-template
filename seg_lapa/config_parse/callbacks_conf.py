from abc import ABC, abstractmethod
from typing import Dict

from omegaconf import DictConfig
from pydantic.dataclasses import dataclass

from seg_lapa.config_parse.conf_utils import asdict_filtered, validate_config_group_generic
from seg_lapa.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass(frozen=True)
class CallbacksConf(ABC):
    name: str

    @abstractmethod
    def get_callbacks_dict(self):
        return {}

    def get_callbacks_list(self, *args):
        callback_dict = self.get_callbacks_dict(*args)
        callback_list = list(callback_dict.values())
        return callback_list


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
class CheckpointConf:
    """Dataclass just to initialize and return the Checkpoint Callback"""

    def get_callback(self, logs_dir):
        checkpoint_callback = ModelCheckpoint(dirpath=logs_dir)
        return checkpoint_callback


@dataclass(frozen=True)
class StandardCallbacksConf(CallbacksConf):
    """Get a dictionary of all the callbacks."""

    early_stopping: Dict

    def get_callbacks_dict(self, logs_dir) -> Dict:
        early_stop = EarlyStopConf(**self.early_stopping).get_callback()
        checkpoint = CheckpointConf().get_callback(logs_dir)
        return {"early_stopping": early_stop, "checkpoint": checkpoint}


valid_names = {
    "disabled": DisabledCallbacksConf,
    "standard": StandardCallbacksConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> CallbacksConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, mapping_names_dataclass=valid_names, config_category="callback"
    )
    return validated_dataclass
