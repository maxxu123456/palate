"""One pydantic model, two dialects. A raw pydantic schema breaks the template one."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from palate.providers.base import JSONObject, SchemaStyle


def render(model: type[BaseModel], *, style: SchemaStyle) -> JSONObject:
    """The argument schema as one dialect wants to receive it."""
    schema: dict[str, Any] = model.model_json_schema()
    schema.pop("title", None)
    if style == "chat_template":
        schema = _inline_defs(schema)
    schema["additionalProperties"] = False
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Splice $defs into their $ref sites. A $ref nothing resolves is noise in the prompt."""
    defs = schema.get("$defs", {})
    if not defs:
        return schema
    out = cast("dict[str, Any]", _walk(schema, lambda node: _deref(node, defs)))
    out.pop("$defs", None)
    return out


def _deref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return node
    target = defs.get(ref.removeprefix("#/$defs/"))
    if not isinstance(target, dict):
        return node
    merged = {k: v for k, v in node.items() if k != "$ref"}
    return {**target, **merged}


def _walk(node: Any, fn: Any) -> Any:
    if isinstance(node, dict):
        return fn({key: _walk(value, fn) for key, value in node.items()})
    if isinstance(node, list):
        return [_walk(item, fn) for item in node]
    return node


def excerpt(schema: JSONObject, field: str) -> JSONObject | None:
    """The schema of one property, for an error that has to show the model what it wanted."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    found = properties.get(field)
    return {field: found} if isinstance(found, dict) else None
