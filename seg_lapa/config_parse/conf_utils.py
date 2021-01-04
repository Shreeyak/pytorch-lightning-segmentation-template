import dataclasses
from typing import Dict, Optional, Sequence

from omegaconf import DictConfig, OmegaConf


def asdict_filtered(obj, remove_keys: Optional[Sequence[str]] = None) -> Dict:
    """Returns the attributes of a dataclass in the form of a dict, with unwanted attributes removed.
    Each config group has the term 'name', which is helpful in identifying the node that was chosen
    in the config group (Eg. config group = optimizers, nodes = adam, sgd).
    However, the 'name' parameter is not required for initializing any dataclasses. Hence it needs to be removed.

    Args:
        obj: The dataclass whose atrributes will be converted to dict
        remove_keys: The keys to remove from the dict. The default is ['name'].
    """
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Not a dataclass/dataclass instance")

    if remove_keys is None:
        remove_keys = ["name"]

    # Clean the arguments
    args = dataclasses.asdict(obj)
    for key in remove_keys:
        if key in args:
            args.pop(key)

    return args


def validate_config_group_generic(cfg_group: DictConfig, dataclass_dict: Dict, config_category: str = "option"):
    """Use a hydra config group to initialize a pydantic dataclass. Initializing it validates the data.
    Each of our config groups has a name parameter, which is used to map to valid dataclasses for validation.

    Pydantic will force the parameters to the desired datatype and will throw errors if the config
    cannot be cast to the dataclass members.

    Args:
        cfg_group: The config group extracted from the hydra config.
        dataclass_dict: A dict containing the mapping from 'name' entry in config files to matching
                        pydantic dataclasses for validation.
        config_category: For pretty print statements. Configure the name of the config group when throwing error.

    Raises:
        ValueError: If the name parameter does not match to any of the valid options
    """
    if not OmegaConf.is_config(cfg_group):
        raise ValueError(f"Given config not an OmegaConf config. Got: {type(cfg_group)}")

    # Get the "name" entry in config
    name = cfg_group.name
    if name is None:
        raise KeyError(
            f"The given config does not contain a 'name' entry. Cannot map to a dataclass.\n"
            f"  Config:\n {OmegaConf.to_yaml(cfg_group)}"
        )

    # Convert hydra config to dict - This dict contains the arguments to init dataclass
    cfg_asdict = OmegaConf.to_container(cfg_group, resolve=True)

    # Get the dataclass to init from the mapping. Init the dataclass using hydra config
    try:
        dataclass_obj = dataclass_dict[name](**cfg_asdict)
    except KeyError:
        raise ValueError(
            f"Invalid Config: '{cfg_group.name}' is not a valid {config_category}. "
            f"Valid Options: {list(dataclass_dict.keys())}"
        )

    return dataclass_obj
