from abc import ABC, abstractmethod
from typing import Dict, List, Optional


from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning.callbacks import Callback

from seg_lapa.config_parse.conf_utils import validate_config_group_generic
from seg_lapa.config_parse.callbacks_available import EarlyStopConf, CheckpointConf, LogMediaConf

# The Callbacks config cannot be directly initialized because it contains sub-entries for each callback, each
# of which describes a separate class.
# For each of the callbacks, we define a dataclass and use them to init the list of callbacks


@dataclass(frozen=True)
class CallbacksConf(ABC):
    name: str

    @abstractmethod
    def get_callbacks_list(self, *args) -> List:
        return []


@dataclass(frozen=True)
class DisabledCallbacksConf(CallbacksConf):
    def get_callbacks_list(self) -> List:
        return []


@dataclass(frozen=True)
class StandardCallbacksConf(CallbacksConf):
    """Get a dictionary of all the callbacks."""

    early_stopping: Optional[Dict] = None
    checkpoints: Optional[Dict] = None
    log_media: Optional[Dict] = None

    def get_callbacks_list(self, logs_dir) -> List[Callback]:
        """Get all available callbacks and the Callback Objects in list
        If a callback's entry is not present in the config file, it'll not be output in the list
        """
        callbacks_list = []
        if self.early_stopping is not None:
            early_stop = EarlyStopConf(**self.early_stopping).get_callback()
            callbacks_list.append(early_stop)

        if self.checkpoints is not None:
            checkpoint = CheckpointConf(**self.checkpoints).get_callback(logs_dir)
            callbacks_list.append(checkpoint)

        if self.log_media is not None:
            log_media = LogMediaConf(**self.log_media).get_callback(logs_dir)
            callbacks_list.append(log_media)

        return callbacks_list


valid_names = {
    "disabled": DisabledCallbacksConf,
    "standard": StandardCallbacksConf,
}


def validate_config_group(cfg_subgroup: DictConfig) -> CallbacksConf:
    validated_dataclass = validate_config_group_generic(
        cfg_subgroup, dataclass_dict=valid_names, config_category="callback"
    )
    return validated_dataclass
