"""One pydantic model, two dialects, and ten tools that have to survive both of them."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import orjson
import pytest
from fixtures.synth import LOVED, pipeline
from fixtures.synth.pipeline import SynthEmbedder

from palate.config import Settings
from palate.providers.base import SchemaStyle, ToolCall
from palate.retrieval.recommend import LocalRecommender
from palate.retrieval.vocab import Vocabulary
from palate.taste.memory import PreferenceStore, ensure_session
from palate.tools.catalog import MODULES, NAMES, build_registry
from palate.tools.context import ToolContext
from palate.tools.envelope import ToolErrorCode, ToolResult
from palate.tools.schema import render

STYLES: tuple[SchemaStyle, ...] = ("plain", "chat_template")

SESSION = "ses_tools"


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pipeline.Fitted]:
    built = pipeline.fit(tmp_path_factory.mktemp("tools"))
    ensure_session(built.db, SESSION)
    yield built
    built.close()


@pytest.fixture
def ctx(fitted: pipeline.Fitted) -> ToolContext:
    """A context over the planted world, with a query embedder in the index's own space."""
    recommender = LocalRecommender(
        fitted.db,
        embedder=SynthEmbedder(fitted.world, cluster=LOVED[0]),
        session_id=SESSION,
        profile=fitted.profile,
    )
    return ToolContext(
        session_id=SESSION,
        run_id="run_tools",
        settings=Settings(),
        db=fitted.db,
        recommender=recommender,
        profile=fitted.profile,
        prefs=PreferenceStore(fitted.db),
        vocab=Vocabulary(fitted.db.read()),
    )


async def call(ctx: ToolContext, name: str, **arguments: Any) -> ToolResult:
    """One dispatch through the real registry, exactly as the loop would do it."""
    registry = build_registry()
    return await registry.dispatch(
        ToolCall(
            id="c1",
            name=name,
            arguments_json=orjson.dumps(arguments).decode(),
            arguments=arguments,
        ),
        ctx,
    )


def conforms(schema: dict[str, Any], value: Any, path: str = "") -> list[str]:
    """A small structural check, enough to catch a dialect that broke a tool."""
    problems: list[str] = []
    options = schema.get("anyOf")
    if isinstance(options, list):
        if not any(not conforms(o, value, path) for o in options):
            problems.append(f"{path} matches no branch of anyOf")
        return problems
    kinds = schema.get("type")
    wanted = kinds if isinstance(kinds, list) else [kinds] if kinds else []
    if wanted and not any(_is(kind, value) for kind in wanted):
        problems.append(f"{path} is {type(value).__name__}, wanted {wanted}")
        return problems
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{path}.{name} is required and missing")
        for name, item in value.items():
            if name not in properties:
                problems.append(f"{path}.{name} is not in the schema")
                continue
            problems.extend(conforms(properties[name], item, f"{path}.{name}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            problems.extend(conforms(schema["items"], item, f"{path}[{index}]"))
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value is not None and value not in allowed:
        problems.append(f"{path} is {value!r}, not one of {allowed}")
    return problems


def _is(kind: str | None, value: Any) -> bool:
    if kind == "null":
        return value is None
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    table = {"string": str, "boolean": bool, "array": list, "object": dict}
    expected = table.get(kind or "")
    return expected is not None and isinstance(value, expected)


def test_the_registry_holds_exactly_ten_tools() -> None:
    registry = build_registry()
    assert len(registry.names()) == 10
    assert registry.names() == NAMES


@pytest.mark.parametrize("style", STYLES)
def test_every_tool_renders_in_every_dialect(style: SchemaStyle) -> None:
    for schema in build_registry().schemas(style=style):
        assert schema.parameters["additionalProperties"] is False
        assert schema.parameters["type"] == "object"
        assert schema.description


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("module", MODULES, ids=[m.SPEC.name for m in MODULES])
def test_every_worked_example_validates_against_its_rendered_schema(
    style: SchemaStyle, module: Any
) -> None:
    schema = render(module.SPEC.args_model, style=style)
    for example in module.SPEC.examples:
        payload = example.model_dump(exclude_none=True)
        assert conforms(schema, payload) == []


def test_the_template_dialect_leaves_no_ref_behind() -> None:
    for schema in build_registry().schemas(style="chat_template"):
        rendered = repr(schema.parameters)
        assert "$ref" not in rendered
        assert "$defs" not in rendered


def test_withdrawing_a_tool_removes_it_from_what_is_offered() -> None:
    registry = build_registry()
    offered = registry.schemas(exclude={"search_films"})
    assert "search_films" not in [s.name for s in offered]
    assert len(offered) == 9
    assert registry.schemas(include={"get_film"})[0].name == "get_film"


async def test_search_films_returns_unwatched_films_with_hooks(ctx: ToolContext) -> None:
    result = await call(ctx, "search_films", query="something slow and cold", limit=5)
    assert result.ok
    assert result.data is not None
    films = result.data["films"]
    assert len(films) == 5
    assert all(f["hook"] for f in films)
    seen = await call(ctx, "check_watched", film_ids=[f["film_id"] for f in films])
    assert seen.data is not None
    assert all(not row["watched"] for row in seen.data["films"])
    assert result.meta["excluded_watched"] > 0


async def test_search_films_says_what_each_clause_removed(ctx: ToolContext) -> None:
    result = await call(
        ctx, "search_films", query="anything at all", exclude_languages=["ru"], limit=5
    )
    assert result.ok
    assert result.meta["removed_by_clause"]["exclude_languages"] > 0


async def test_an_unknown_language_comes_back_with_the_corpus_vocabulary(
    ctx: ToolContext,
) -> None:
    result = await call(
        ctx, "search_films", query="anything at all", exclude_languages=["Sovietish"]
    )
    assert not result.ok
    assert result.error is not None
    assert result.error.code is ToolErrorCode.BAD_ARGUMENTS
    assert "Sovietish" in result.error.hint


async def test_filter_films_needs_no_embedding_and_still_excludes_watched(
    ctx: ToolContext,
) -> None:
    result = await call(ctx, "filter_films", exclude_languages=["ru"], limit=8)
    assert result.ok
    assert result.data is not None
    assert result.meta["excluded_watched"] > 0
    assert all(f["original_language"] != "ru" for f in result.data["films"])


async def test_filter_films_can_sort_by_something_other_than_taste(ctx: ToolContext) -> None:
    result = await call(ctx, "filter_films", order_by="year_desc", limit=6)
    assert result.ok
    assert result.data is not None
    years = [f["year"] for f in result.data["films"]]
    assert years == sorted(years, reverse=True)


async def test_get_film_returns_the_overview_and_where_it_sits(ctx: ToolContext) -> None:
    listed = await call(ctx, "filter_films", limit=1)
    assert listed.data is not None
    film_id = listed.data["films"][0]["film_id"]
    result = await call(ctx, "get_film", film_ids=[film_id])
    assert result.ok
    assert result.data is not None
    record = result.data["films"][0]
    assert record["film_id"] == film_id
    assert record["overview_length"] == len(record["overview"])
    assert record["overview_offset"] >= 0


async def test_get_film_refuses_an_id_the_corpus_does_not_hold(ctx: ToolContext) -> None:
    result = await call(ctx, "get_film", film_ids=[424242])
    assert not result.ok
    assert result.error is not None
    assert result.error.code is ToolErrorCode.NOT_FOUND


async def test_compare_films_returns_set_algebra_and_never_prose(ctx: ToolContext) -> None:
    listed = await call(ctx, "filter_films", limit=2)
    assert listed.data is not None
    a, b = (f["film_id"] for f in listed.data["films"])
    result = await call(ctx, "compare_films", film_id_a=a, film_id_b=b)
    assert result.ok
    assert result.data is not None
    assert result.data["a"]["film_id"] == a
    assert isinstance(result.data["shared_keywords"], list)
    assert isinstance(result.data["same_language"], bool)


async def test_get_taste_profile_carries_support_counts_beside_every_claim(
    ctx: ToolContext,
) -> None:
    result = await call(ctx, "get_taste_profile")
    assert result.ok
    assert result.data is not None
    assert result.data["tier"] == "full"
    assert result.data["modes"]
    assert all(m["n_members"] > 0 for m in result.data["modes"])
    assert all(d["n"] > 0 for d in result.data["top_directors"])


async def test_get_ratings_is_the_only_place_the_users_own_stars_come_from(
    ctx: ToolContext,
) -> None:
    result = await call(ctx, "get_ratings", rating_min=4.0, limit=10)
    assert result.ok
    assert result.data is not None
    assert result.data["ratings"]
    assert all(r["your_rating"] >= 4.0 for r in result.data["ratings"])


async def test_check_watched_catches_a_title_that_does_not_exist(ctx: ToolContext) -> None:
    result = await call(ctx, "check_watched", titles=["A Film Nobody Made"])
    assert result.ok
    assert result.data is not None
    assert result.data["films"][0]["in_corpus"] is False
    assert result.meta["not_in_corpus"] == ["A Film Nobody Made"]


async def test_resolve_vocabulary_turns_a_nationality_into_a_language(ctx: ToolContext) -> None:
    result = await call(ctx, "resolve_vocabulary", text="Russian")
    assert result.ok
    assert result.data is not None
    kinds = {m["kind"] for m in result.data["matches"]}
    assert "language" in kinds or "country" in kinds
    assert all(m["affected_films"] >= 0 for m in result.data["matches"])


async def test_a_tool_with_no_database_behind_it_says_so_rather_than_crashing() -> None:
    empty = ToolContext(session_id="ses_empty", run_id="run_empty", settings=Settings())
    result = await call(empty, "get_taste_profile")
    assert not result.ok
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PRECONDITION_FAILED
