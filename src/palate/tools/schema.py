"""One pydantic model, four dialects. A raw pydantic schema breaks two of them."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from palate.providers.base import JSONObject, SchemaStyle

_NULL = {"type": "null"}


def render(model: type[BaseModel], *, style: SchemaStyle) -> JSONObject:
    """The argument schema as one provider dialect wants to receive it."""
    schema: dict[str, Any] = model.model_json_schema()
    schema.pop("title", None)
    if style == "ollama":
        schema = _inline_defs(schema)
    if style == "openai_strict":
        schema = _strictify(schema)
    if style == "hf":
        schema = _flatten_enums(schema)
    schema["additionalProperties"] = False
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Splice $defs into their $ref sites. Ollama and several local servers ignore $ref."""
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


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode wants every property in required, so optionals become nullable."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema
    required = set(schema.get("required", []))
    out = dict(schema)
    out["properties"] = {
        name: (node if name in required else _nullable(node)) for name, node in properties.items()
    }
    out["required"] = sorted(properties)
    return out


def _nullable(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    options = node.get("anyOf")
    if isinstance(options, list):
        kinds = [o.get("type") for o in options if isinstance(o, dict)]
        real = [k for k in kinds if isinstance(k, str) and k != "null"]
        if real and len(real) == len(kinds) - kinds.count("null"):
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            body = next(o for o in options if isinstance(o, dict) and o.get("type") != "null")
            return {**body, **rest, "type": [*real, "null"]}
    kind = node.get("type")
    if isinstance(kind, str):
        return {**node, "type": [kind, "null"]}
    if "anyOf" in node:
        return {**node, "anyOf": [*node["anyOf"], _NULL]}
    return node


def _flatten_enums(schema: dict[str, Any]) -> dict[str, Any]:
    """Enum values as plain strings, which is what the HF router accepts."""

    def flatten(node: dict[str, Any]) -> dict[str, Any]:
        values = node.get("enum")
        if not isinstance(values, list):
            return node
        return {**node, "enum": [str(v) for v in values]}

    return cast("dict[str, Any]", _walk(schema, flatten))


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
