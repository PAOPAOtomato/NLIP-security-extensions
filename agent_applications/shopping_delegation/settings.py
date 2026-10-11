"""Location and loading of the shopping-delegation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nlip_sdk.config import load_config

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.toml"


def load_settings(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_config(path)


def resolve_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the application directory."""
    path = Path(path)
    return path if path.is_absolute() else APP_DIR / path
