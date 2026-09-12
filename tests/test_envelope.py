"""One shape for every tool result, and a truncation the model can resume from."""

from __future__ import annotations

import orjson
import pytest

from palate.tools.envelope import (
    RETRYABLE,
    ToolError,
    ToolErrorCode,
    ToolResult,
    failure,
)

BIG = {"films": [{"film_id": n, "title": f"film number {n}"} for n in range(200)]}


def ok(**fields: object) -> ToolResult:
    return ToolResult(call_id="c1", name="search_films", ok=True, **fields)  # type: ignore[arg-type]


def test_a_success_renders_as_ok_data_and_meta() -> None:
    body = orjson.loads(ok(data={"films": []}, meta={"count": 0}).render(4000))
    assert body == {"ok": True, "data": {"films": []}, "meta": {"count": 0}}


def test_a_failure_renders_the_code_the_message_and_the_hint() -> None:
    result = failure(
        "c1",
        "search_films",
        ToolErrorCode.BAD_ARGUMENTS,
        "year_min must be <= year_max",
        hint="you passed year_min=1990, year_max=1975",
    )
    body = orjson.loads(result.render(4000))
    assert body["ok"] is False
    assert body["error"]["code"] == "bad_arguments"
    assert body["error"]["retryable"] is True
    assert "1975" in body["error"]["hint"]


def test_truncation_is_announced_with_an_offset_to_resume_from() -> None:
    rendered = ok(data=BIG).render(500)
    assert len(rendered) <= 500
    assert '"truncated": true' in rendered
    assert '"offset": 500' in rendered


def test_a_result_that_fits_is_not_touched() -> None:
    result = ok(data={"films": []})
    assert result.render(4000) == result.render(10_000)


@pytest.mark.parametrize("code", list(ToolErrorCode))
def test_every_code_declares_whether_trying_again_could_work(code: ToolErrorCode) -> None:
    result = failure("c1", "search_films", code, "something")
    assert result.error is not None
    assert result.error.retryable is (code in RETRYABLE)


def test_an_unrecoverable_code_is_not_marked_retryable() -> None:
    for code in (ToolErrorCode.REFUSED, ToolErrorCode.REPEATED_CALL, ToolErrorCode.UNKNOWN_TOOL):
        result = failure("c1", "search_films", code, "no")
        assert result.error is not None
        assert not result.error.retryable


def test_the_summary_is_one_line_and_never_the_payload() -> None:
    assert ok(data=BIG, meta={"returned": 200}).summary() == "search_films returned 200"
    broken = failure("c1", "search_films", ToolErrorCode.TIMEOUT, "slow")
    assert broken.summary() == "search_films failed: timeout"


def test_valid_values_and_the_schema_excerpt_ride_along_when_there_are_any() -> None:
    result = failure(
        "c1",
        "search_films",
        ToolErrorCode.BAD_ARGUMENTS,
        "unknown language",
        schema_excerpt={"exclude_languages": {"type": "array"}},
        valid_values=("ru", "uk", "ka"),
    )
    body = orjson.loads(result.render(4000))
    assert body["error"]["valid_values"] == ["ru", "uk", "ka"]
    assert body["error"]["schema"]["exclude_languages"]["type"] == "array"


def test_reshaping_an_error_keeps_the_call_it_belongs_to() -> None:
    first = failure("c9", "get_film", ToolErrorCode.BAD_ARGUMENTS, "nope", hint="try again")
    assert first.error is not None
    louder = first.with_error(
        ToolError(code=first.error.code, message=first.error.message, retryable=True, hint="louder")
    )
    assert louder.call_id == "c9"
    assert louder.name == "get_film"
    assert louder.error is not None
    assert louder.error.hint == "louder"
    assert not louder.ok


def test_an_empty_meta_is_left_out_rather_than_sent_as_an_empty_object() -> None:
    assert "meta" not in orjson.loads(ok(data={"films": []}).render(4000))
