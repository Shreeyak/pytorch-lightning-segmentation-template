import dataclasses
from typing import Optional, Sequence, Dict


def cleaned_asdict(obj, remove_keys: Optional[Sequence[str]] = None) -> Dict:
    """Returns the attributes of a dataclass in the form of a dict, with unwanted attributes removed
    Each config group has the term 'name', which is not required in initializing any classes. So it needs to be removed.

    Args:
        obj: The dataclass whose atrributes will be converted to dict
        remove_keys: The keys to remove from the dict. The default is ['name'].
    """
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f'Not a dataclass/dataclass instance')

    if remove_keys is None:
        remove_keys = ['name']

    # Clean the arguments
    args = dataclasses.asdict(obj)
    for key in remove_keys:
        args.pop(key)

    return args
