from pathlib import Path
from typing import Union, Optional


def get_project_root() -> Path:
    """Get the root dir of the project (one level above package).
    Used to get paths relative to the project, mainly for placing log files"""
    return Path(__file__).parent.parent.parent


def get_path(input_path: Union[str, Path], path_type: str = 'dir', must_exist: Optional[bool] = None,
             force_relative_to_project: bool = False) -> Path:
    """Converts a str to a pathlib Path with added checks

    Args:
        input_path: The path to be converted
        path_type: Whether the path is to a dir or file.
                   Options: ['dir', 'file']
        must_exist: If given, ensure that the path either exists or does not exist.
                    - None: Don't check
                    - True: It must exist.
                    - False: It must not exist.
        force_relative_to_project: If a relative path is given, should it be forced to resolve to a path that's relative
                                   to the project. Useful to always place logs in the project root,
                                   regardless of where the package is called from.

    """
    valid_pathtypes = ['dir', 'file']
    if path_type not in valid_pathtypes:
        raise ValueError(f"Invalid path type '{path_type}'. Valid options: {valid_pathtypes}")

    input_path = Path(input_path)

    if not input_path.expanduser().is_absolute() and force_relative_to_project:
        # If the input path is relative, change it to relative to the project root.
        proj_root = get_project_root()
        input_path = proj_root / input_path

    if must_exist is not None:
        if must_exist:
            # check that path exists and is the correct type
            if not input_path.exists():
                raise ValueError(f"Could not find {path_type}. Does not exist: {input_path}")

            if not input_path.is_dir() and path_type == 'dir':
                raise ValueError(f"Not a dir: {input_path}")

            if not input_path.is_file() and path_type == 'file':
                raise ValueError(f"Not a file: {input_path}")
        else:
            # Ensure path doesn't already exist
            if input_path.exists():
                raise ValueError(f"Path already exists: {input_path}")

    return input_path
