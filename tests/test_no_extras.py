"""The base install is the whole app. Only the http surface is still a choice."""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import pytest

import palate
from palate.extras import have

# A web framework cannot be lazily imported, so these modules exist only with the extra.
NEEDS_EXTRA = {"palate.api": "fastapi"}

# Loading a checkpoint costs seconds, so `palate --help` must not reach any of them.
HEAVY = ("torch", "transformers", "sentence_transformers")


def installed(name: str) -> bool:
    return all(not name.startswith(prefix) or have(mod) for prefix, mod in NEEDS_EXTRA.items())


def module_names() -> list[str]:
    found = [m.name for m in pkgutil.walk_packages(palate.__path__, "palate.")]
    return [name for name in found if installed(name)]


def test_there_is_something_to_walk() -> None:
    names = module_names()
    assert "palate.providers.embed.sentence_transformers" in names
    assert ("palate.api.app" in names) == have("fastapi")
    assert len(names) > 30


@pytest.mark.parametrize("name", module_names())
def test_every_module_imports_on_its_own(name: str) -> None:
    importlib.import_module(name)


def test_the_cli_does_not_load_a_model_stack_to_print_help() -> None:
    code = (
        "import sys, importlib; importlib.import_module('palate.cli'); "
        f"print([h for h in {HEAVY!r} if h in sys.modules])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
