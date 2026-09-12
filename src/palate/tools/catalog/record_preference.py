"""record_preference: the only write, so the quote is checked against what the user said."""

from __future__ import annotations

import sqlite3
from functools import partial
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field

from palate.errors import ToolFailure
from palate.retrieval.vocab import Vocabulary
from palate.taste.memory import PreferenceDraft, PreferenceStore
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Save a preference the user stated in this conversation. evidence_quote must be an exact "
    "substring of something the user actually typed. Resolve the target with resolve_vocabulary "
    "first. Every write is shown to the user and can be undone."
)

# A preference attached to one film is never generalised to its genre.
NEVER_DURABLE = frozenset({"film"})

DURABLE_STRENGTH = 2

_VOCAB_KIND = {
    "genre": "genre",
    "keyword": "keyword",
    "language": "language",
    "country": "country",
    "collection": "collection",
    "director": "person",
    "actor": "person",
    "writer": "person",
}

_BY_TITLE = (
    "select tmdb_id, title from films where title = ? collate nocase "
    "order by vote_count desc limit 1"
)

type TargetKind = Literal[
    "genre",
    "keyword",
    "director",
    "actor",
    "writer",
    "country",
    "language",
    "decade",
    "runtime",
    "film",
    "collection",
    "freeform",
]


class RecordPreferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    polarity: Literal["like", "dislike"]
    target_kind: TargetKind
    target: str = Field(min_length=1, max_length=120)
    strength: int = Field(ge=1, le=3)
    hardness: Literal["hard", "soft"]
    scope: Literal["session", "durable"] = "session"
    evidence_quote: str = Field(min_length=3, max_length=300)


class RecordPreferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pref_id: int
    target_kind: str
    target_id: str
    target_label: str
    polarity: str
    strength: int
    hardness: str
    scope: str
    resolved_ids: list[str] = Field(default_factory=list)
    affected_films: int = 0
    undo_token: str
    meta: dict[str, object] = Field(default_factory=dict)


def resolve_target(
    vocab: Vocabulary, conn: sqlite3.Connection, kind: str, target: str
) -> tuple[str, str, tuple[str, ...], int]:
    """The corpus ids a stated target means, so the row says what it actually excludes."""
    if kind in ("decade", "runtime"):
        digits = "".join(c for c in target if c.isdigit())
        if not digits:
            raise ToolFailure("bad_arguments", f"{kind} preference {target!r} names no number")
        return digits, target, (digits,), 0
    if kind == "film":
        row = conn.execute(_BY_TITLE, (target,)).fetchone()
        if row is None:
            raise ToolFailure(
                "bad_arguments",
                f"no film in the corpus is called {target!r}",
                hint="check the title with check_watched before recording a preference about it",
            )
        return str(row["tmdb_id"]), str(row["title"]), (str(row["tmdb_id"]),), 1
    wanted = _VOCAB_KIND.get(kind)
    if wanted is None:
        return target.casefold().replace(" ", "-"), target, (), 0
    found = [m for m in vocab.resolve(target, prefer=[wanted]) if m.kind == wanted]
    if not found:
        close = vocab.suggest(wanted, target)
        listed = ", ".join(f"{label} ({films} films)" for _, label, films in close)
        raise ToolFailure(
            "bad_arguments",
            f"nothing in this corpus matches {target!r} as a {kind}",
            hint=f"closest: {listed}" if listed else "call resolve_vocabulary first",
            valid_values=tuple(label for _, label, _ in close),
        )
    best = found[0]
    return best.ids[0], best.label, best.ids, best.affected_films


def scope_for(args: RecordPreferenceArgs) -> str:
    """Durable memory has to be earned. Anything weaker stays in this session."""
    if args.scope != "durable":
        return "session"
    if args.strength < DURABLE_STRENGTH or args.target_kind in NEVER_DURABLE:
        return "session"
    return "durable"


def write(
    store: PreferenceStore,
    vocab: Vocabulary,
    args: RecordPreferenceArgs,
    ctx: ToolContext,
) -> RecordPreferenceResult:
    """Resolve, then write. PreferenceStore refuses a quote nobody in this session said."""
    conn = store.db.read()
    target_id, label, ids, affected = resolve_target(vocab, conn, args.target_kind, args.target)
    scope = scope_for(args)
    stored = store.record(
        PreferenceDraft(
            target_kind=args.target_kind,
            target_id=target_id,
            target_label=label,
            polarity=args.polarity,
            strength=args.strength,
            hardness=args.hardness,
            evidence_quote=args.evidence_quote,
            resolved_ids=ids,
            affected_films=affected,
            scope="durable" if scope == "durable" else "session",
        ),
        session_id=ctx.session_id,
        user_messages=ctx.user_messages,
    )
    return RecordPreferenceResult(
        pref_id=stored.pref_id,
        target_kind=stored.target_kind,
        target_id=stored.target_id,
        target_label=stored.target_label,
        polarity=stored.polarity,
        strength=stored.strength,
        hardness=stored.hardness,
        scope=stored.scope,
        resolved_ids=list(stored.resolved_ids),
        affected_films=stored.affected_films,
        undo_token=f"pref:{stored.pref_id}",
        meta={
            "count": 1,
            "returned": 1,
            "scope_downgraded": scope != args.scope,
            "compiles_to_filter": stored.is_hard,
        },
    )


async def handler(args: RecordPreferenceArgs, ctx: ToolContext) -> RecordPreferenceResult:
    """One short transaction, off the event loop, never holding the lock across an await."""
    ctx.check_deadline()
    store = ctx.require_prefs()
    vocab = ctx.require_vocab()
    return await anyio.to_thread.run_sync(partial(write, store, vocab, args, ctx))


SPEC = ToolSpec(
    name="record_preference",
    description=DESCRIPTION,
    args_model=RecordPreferenceArgs,
    result_model=RecordPreferenceResult,
    kind=ToolKind.WRITE,
    handler=handler,
    examples=(
        RecordPreferenceArgs(
            polarity="dislike",
            target_kind="keyword",
            target="musical",
            strength=3,
            hardness="hard",
            scope="durable",
            evidence_quote="I hate musicals",
        ),
    ),
    cacheable=False,
    cost_hint_ms=40,
    max_result_chars=1500,
    invalidates=frozenset({"search_films", "filter_films", "list_preferences"}),
)
