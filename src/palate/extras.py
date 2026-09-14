"""Lazy-import guard for the optional dependency groups."""

from __future__ import annotations

import importlib
from types import ModuleType

from palate.errors import MissingExtra


def require(extra: str, module: str) -> ModuleType:
    """Import a module from an optional extra, raising MissingExtra if it is absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(extra, module) from exc


def have(module: str) -> bool:
    """True when the module can be imported. Does not raise."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
