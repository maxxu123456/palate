"""The ten tools, and the one function that puts them in a registry."""

from __future__ import annotations

from palate.tools.catalog import (
    check_watched,
    compare_films,
    filter_films,
    get_film,
    get_ratings,
    get_taste_profile,
    list_preferences,
    record_preference,
    resolve_vocabulary,
    search_films,
)
from palate.tools.registry import ToolRegistry

# Order is the order the model sees them in, so the two search tools sit next to each other.
MODULES = (
    search_films,
    filter_films,
    get_film,
    compare_films,
    get_taste_profile,
    get_ratings,
    check_watched,
    resolve_vocabulary,
    record_preference,
    list_preferences,
)

NAMES = tuple(module.SPEC.name for module in MODULES)


def build_registry() -> ToolRegistry:
    """Every tool the agent may call."""
    registry = ToolRegistry()
    for module in MODULES:
        registry.register(module.SPEC)
    return registry
