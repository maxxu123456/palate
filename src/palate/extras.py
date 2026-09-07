"""Lazy-import guard for the optional dependency groups."""

from __future__ import annotations

import importlib
from types import ModuleType

from palate.errors import MissingExtra

# Which extra installs which module, so the error can name the right uv command.
_EXTRA_FOR: dict[str, str] = {
    "torch": "local",
    "sentence_transformers": "local",
    "transformers": "local",
    "huggingface_hub": "hf",
    "fastapi": "api",
    "uvicorn": "api",
    "sse_starlette": "api",
    "scipy": "eval",
    "matplotlib": "eval",
}


def require(extra: str, module: str) -> ModuleType:
    """Import a module from an optional extra, raising MissingExtra if it is absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(extra, module) from exc


def require_module(module: str) -> ModuleType:
    """Same, looking the extra up from the module name."""
    return require(_EXTRA_FOR.get(module.split(".")[0], "all"), module)


def have(module: str) -> bool:
    """True when the module can be imported. Does not raise."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
