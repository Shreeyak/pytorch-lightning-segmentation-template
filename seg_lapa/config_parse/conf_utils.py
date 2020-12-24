import dataclasses
from typing import Optional, Sequence, Dict

from omegaconf import DictConfig


def cleaned_asdict(obj, remove_keys: Optional[Sequence[str]] = None) -> Dict:
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
        args.pop(key)

    return args


def validate_config_group_generic(
    cfg_group: DictConfig, mapping_names_dataclass: Dict, config_category: Optional[str] = None
):
    """Validate a hydra config group (DictConfig) by using it to initialize a pydantic dataclass.
    Each of our config groups has a name parameter, which is used to map to valid dataclasses for validation.

    Pydantic will force the parameters to the desired datatype and will throw errors if the config
    cannot be cast to the dataclass members.

    Args:
        cfg_group: The config group extracted from the hydra config.
        mapping_names_dataclass: A dict containing the mapping from 'name' entry in config to matching
                                 pydantic dataclasses for validation.
        config_category: For pretty print statements. Configure the name of the config group when throwing error.

    Raises:
        ValueError: If the name parameter does not match to any of the valid options
    """
    try:
        # Get the dataclass to init from the mapping
        dataclass_config = mapping_names_dataclass[cfg_group.name]

        # Init the dataclass using hydra config
        dataconf = dataclass_config(**cfg_group)

    except KeyError:
        if config_category is None:
            config_category = "option"

        raise ValueError(
            f"Invalid Config: '{cfg_group.name}' is not a valid {config_category}. "
            f"Valid Options: {list(mapping_names_dataclass.keys())}"
        )

    return dataconf
