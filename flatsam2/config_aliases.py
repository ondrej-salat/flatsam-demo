"""Utilities for loading display aliases for tracker config names."""

import importlib.util
from pathlib import Path
from typing import Mapping, Optional


def load_config_aliases_from_file(alias_config_path: Optional[Path]) -> dict[str, str]:
    """Load config display aliases from a Python file.

    Args:
        alias_config_path: Path to a Python alias config file. If ``None``,
            returns an empty mapping.

    Returns:
        Mapping from config directory names (e.g. ``"flow_hough"``) to display
        aliases (e.g. ``"Flow+Hough"``).

    Raises:
        FileNotFoundError: If the provided alias config path does not exist.
        ValueError: If the file cannot be loaded or does not define valid alias
            mapping data.
    """
    if alias_config_path is None:
        return {}

    alias_config_path = Path(alias_config_path)
    if not alias_config_path.exists():
        raise FileNotFoundError(f"Alias config file not found: {alias_config_path}")

    spec = importlib.util.spec_from_file_location(
        "config_aliases_module", alias_config_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load alias config from {alias_config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_aliases = getattr(module, "aliases", None)
    if raw_aliases is None:
        raw_aliases = getattr(module, "config_aliases", None)
    if raw_aliases is None:
        raise ValueError(
            f"Alias config file {alias_config_path} must define 'aliases' "
            f"(or legacy 'config_aliases')"
        )
    if not isinstance(raw_aliases, Mapping):
        raise ValueError(
            f"Alias config in {alias_config_path} must be a mapping, got {type(raw_aliases)}"
        )

    aliases: dict[str, str] = {}
    for key, value in raw_aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"Alias entries must be str->str, got key={type(key)}, value={type(value)}"
            )
        aliases[key] = value

    return aliases


def get_config_display_name(config_name: str, aliases: Mapping[str, str]) -> str:
    """Return aliased display name for a config directory name."""
    return aliases.get(config_name, config_name)
