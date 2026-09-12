"""What happens when the model gets it wrong, and why the run still answers."""

from __future__ import annotations

from collections.abc import Sequence

from fixtures.agent import BUDGET, Calls, build_registry, call, context_factory, refuse

from palate.agent.loop import APOLOGY, AgentLoop, RunOutcome
from palate.agent.prompts import PromptRegistry
from palate.agent.state import StopReason
from palate.agent.transcript import Transcript
from palate.config import Settings
from palate.db.connect import Database
from palate.errors import ProviderContextOverflow, ProviderUnavailable
from palate.providers.base import ChatCapabilities, Message, ToolCall, ToolSchema
from palate.providers.chat.fake import FakeChatProvider, ScriptedTurn
from palate.tools.envelope import ToolErrorCode

ANSWER = "Here is what I could verify."

BROKEN = ToolCall(id="c1", name="search_films", arguments_json="{query: slow", arguments=None)

WRONG = call("search_films", '{"query": "slow", "decade": 1970}')

NARROW = call("search_films", '{"query": "slow"}')

_SHORT = ("a", "ab", "b")


def ask(n: int) -> str:
    """A distinct valid query, so repeat detection does not fire on the fan out."""
    return f'{{"query": "ask {n}"}}'


def short(n: int) -> ToolCall:
    """Too short to validate, and distinct, so repeat detection does not mask the strike."""
    return call("search_films", f'{{"query": "{_SHORT[n]}"}}', ident=f"s{n}")


async def drive(
    turns: list[ScriptedTurn],
    seen: Calls,
    *,
    capabilities: ChatCapabilities | None = None,
    loop_last: bool = False,
    fail: object = None,
    transcript: Transcript | None = None,
) -> RunOutcome:
    """One run, with whatever the scripted provider is told to do to it."""
    provider = FakeChatProvider(turns, capabilities=capabilities, loop_last=loop_last)
    registry = build_registry(seen, fail=fail)  # type: ignore[arg-type]
    agent = AgentLoop(provider, registry, PromptRegistry(), Settings(), transcript=transcript)
    return await agent.run_to_completion(
        "something slow and cold",
        session_id="ses_recovery",
        budget=BUDGET,
        ctx_factory=context_factory(),
    )


async def test_malformed_json_never_reaches_pydantic_and_shows_the_model_its_own_text() -> None:
    seen = Calls()
    outcome = await drive([ScriptedTurn(tool_calls=(BROKEN,)), ScriptedTurn(text=ANSWER)], seen)
    error = outcome.results[0].error
    assert error is not None
    assert error.code is ToolErrorCode.MALFORMED_JSON
    assert "{query: slow" in error.hint
    assert outcome.text == ANSWER


async def test_an_invented_parameter_is_answered_with_the_real_ones() -> None:
    seen = Calls()
    outcome = await drive([ScriptedTurn(tool_calls=(WRONG,)), ScriptedTurn(text=ANSWER)], seen)
    error = outcome.results[0].error
    assert error is not None
    assert error.code is ToolErrorCode.BAD_ARGUMENTS
    assert "decade is not a parameter of search_films" in error.hint
    assert "query" in error.hint and "limit" in error.hint


async def test_an_unknown_tool_name_gets_the_closest_real_one() -> None:
    seen = Calls()
    typo = call("search_film", '{"query": "slow"}')
    outcome = await drive([ScriptedTurn(tool_calls=(typo,)), ScriptedTurn(text=ANSWER)], seen)
    error = outcome.results[0].error
    assert error is not None
    assert error.code is ToolErrorCode.UNKNOWN_TOOL
    assert "search_films" in error.hint
    assert "search_films" in error.valid_values


async def test_the_second_strike_carries_a_worked_call_and_the_third_withdraws_the_tool() -> None:
    seen = Calls()
    offered: list[tuple[str, ...]] = []

    def watch(messages: Sequence[Message], tools: Sequence[ToolSchema]) -> None:
        offered.append(tuple(t.name for t in tools))

    turns = [ScriptedTurn(tool_calls=(short(n),), expect=watch) for n in range(3)]
    turns.append(ScriptedTurn(text=ANSWER, expect=watch))
    outcome = await drive(turns, seen)
    second = outcome.results[1].error
    assert second is not None
    assert "a call that works" in second.hint
    assert second.schema_excerpt is not None
    third = outcome.results[2].error
    assert third is not None
    assert third.hint == "do not call search_films again in this run, answer with what you have"
    assert "search_films" in offered[2]
    assert "search_films" not in offered[3]


async def test_three_failing_calls_in_a_row_end_the_run_with_an_answer() -> None:
    seen = Calls()
    turns = [
        ScriptedTurn(tool_calls=(call("search_films", ask(n), ident=f"c{n}"),)) for n in range(3)
    ]
    outcome = await drive([*turns, ScriptedTurn(text=ANSWER)], seen, fail=refuse)
    assert outcome.stop_reason is StopReason.TOOL_ERROR_STREAK
    assert outcome.text == ANSWER
    assert outcome.state.consecutive_tool_errors >= 3


async def test_a_fan_out_past_the_cap_runs_the_first_four_and_refuses_the_rest() -> None:
    seen = Calls()
    many = tuple(call("search_films", ask(n), ident=f"c{n}") for n in range(6))
    outcome = await drive([ScriptedTurn(tool_calls=many), ScriptedTurn(text=ANSWER)], seen)
    assert len(seen.searches) == 4
    refused = [r for r in outcome.results if not r.ok]
    assert len(refused) == 2
    assert all(
        r.error is not None and r.error.code is ToolErrorCode.TOO_MANY_CALLS for r in refused
    )
    assert refused[0].error is not None
    assert "at most 4 at a time" in refused[0].error.hint


async def test_a_model_without_parallel_calls_keeps_one_per_tool_name() -> None:
    seen = Calls()
    pair = (
        call("search_films", '{"query": "one"}', ident="a"),
        call("search_films", '{"query": "two"}', ident="b"),
    )
    outcome = await drive(
        [ScriptedTurn(tool_calls=pair), ScriptedTurn(text=ANSWER)],
        seen,
        capabilities=ChatCapabilities(parallel_tool_calls=False, context_window=8192),
    )
    assert seen.searches == ["one"]
    error = outcome.results[1].error
    assert error is not None
    assert error.code is ToolErrorCode.UNSUPPORTED
    assert "one call per tool per turn" in error.hint


async def test_the_third_identical_call_is_a_hard_error_and_ends_the_run() -> None:
    seen = Calls()
    turns = [
        ScriptedTurn(tool_calls=(call("search_films", '{"query": "same"}', ident=f"c{n}"),))
        for n in range(3)
    ]
    outcome = await drive([*turns, ScriptedTurn(text=ANSWER)], seen)
    assert seen.searches == ["same"]
    assert outcome.results[1].cache_hit
    error = outcome.results[2].error
    assert error is not None
    assert error.code is ToolErrorCode.REPEATED_CALL
    assert outcome.stop_reason is StopReason.REPEAT_LOOP
    assert outcome.text == ANSWER


async def test_a_provider_failure_while_planning_still_produces_an_answer() -> None:
    seen = Calls()
    outcome = await drive(
        [
            ScriptedTurn(raises=ProviderUnavailable("down", provider="fake")),
            ScriptedTurn(text=ANSWER),
        ],
        seen,
    )
    assert outcome.stop_reason is StopReason.PROVIDER_FAILED
    assert outcome.text == ANSWER


async def test_a_failing_force_answer_apologises_rather_than_raising() -> None:
    seen = Calls()
    outcome = await drive(
        [ScriptedTurn(raises=ProviderUnavailable("down", provider="fake"))],
        seen,
        loop_last=True,
    )
    assert outcome.failed
    assert outcome.text == APOLOGY
    failures = [e for e in outcome.events if type(e).__name__ == "RunFailed"]
    assert len(failures) == 1


async def test_a_context_overflow_compacts_once_and_a_second_one_gives_up(
    tmp_path: object,
) -> None:
    seen = Calls()
    from palate.db.connect import open_database
    from palate.paths import migrations_dir

    db: Database = open_database(
        tmp_path / "palate.db",  # type: ignore[operator]
        migrations=migrations_dir(),
        load_vec=False,
    )
    try:
        outcome = await drive(
            [
                ScriptedTurn(raises=ProviderContextOverflow("too long", provider="fake")),
                ScriptedTurn(tool_calls=(NARROW,)),
                ScriptedTurn(text=ANSWER),
            ],
            seen,
            transcript=Transcript(db),
        )
    finally:
        db.close()
    assert outcome.state.compactions == 1
    assert outcome.text == ANSWER
    assert seen.searches == ["slow"]
