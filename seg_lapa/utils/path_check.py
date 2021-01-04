from enum import Enum
from pathlib import Path
from typing import Optional, Union


class PathType(Enum):
    FILE = 0
    DIR = 1
    ANY = 2


def get_project_root() -> Path:
    """Get the root dir of the project (one level above package).
    Used to get paths relative to the project, mainly for placing log files

    This function assumes that this module is at project_root/package_dir/utils_dir/module.py
    If this structure changes, the func must change.
    """
    return Path(__file__).parent.parent.parent


def get_path(
    input_path: Union[str, Path],
    must_exist: Optional[bool] = None,
    path_type: PathType = PathType.DIR,
    force_relative_to_project: bool = False,
) -> Path:
    """Converts a str to a pathlib Path with added checks

    Args:
        input_path: The path to be converted
        must_exist: If given, ensure that the path either exists or does not exist.
                    - None: Don't check
                    - True: It must exist.
                    - False: It must not exist.
        path_type: Whether the path is to a dir or file. Only used if must_exist is not None.
        force_relative_to_project: If a relative path is given, convert it to be relative to the project root dir.
                                   Required because the package can be called from anywhere, and the relative path will
                                   be relative to where it's called from, rather than relative to the script location.

                                   Useful to always place logs in the project root, regardless of where the package
                                   is called from.
    """
    if not isinstance(path_type, PathType):
        raise ValueError(f"Invalid path type '{path_type} (type={type(path_type)})'. Must be of type Enum {PathType}")

    input_path = Path(input_path)

    if not input_path.expanduser().is_absolute() and force_relative_to_project:
        # If the input path is relative, change it to become relative to the project root.
        proj_root = get_project_root()
        input_path = proj_root / input_path

    if must_exist is not None:
        if must_exist:
            # check that path exists
            if not input_path.exists():
                raise ValueError(f"Could not find {path_type.name.lower()}. Does not exist: {input_path}")

            # If required, check that it is the correct type (file vs dir)
            if path_type is not PathType.ANY:
                if not input_path.is_dir() and path_type == PathType.DIR:
                    raise ValueError(f"Not a dir: {input_path}")

                if not input_path.is_file() and path_type == PathType.FILE:
                    raise ValueError(f"Not a file: {input_path}")
        else:
            # Ensure path doesn't already exist
            if input_path.exists():
                raise ValueError(f"Path already exists: {input_path}")

    return input_path
