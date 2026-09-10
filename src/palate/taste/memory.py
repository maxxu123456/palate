"""Stated preferences as rows, because a preference that only changes the prompt changes nothing."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import orjson

from palate.clock import now_iso
from palate.db.connect import Database
from palate.errors import PreferenceRefused
from palate.providers.base import Message

type Polarity = Literal["like", "dislike"]
type Hardness = Literal["hard", "soft"]
type Scope = Literal["session", "durable"]
type Source = Literal["agent", "user", "inferred"]

# Only an absolute hard preference compiles to SQL. Everything softer becomes a score penalty.
HARD_STRENGTH = 3

PENALTY_PER_STRENGTH = 0.5

QUOTE_HINT = "quote not found in this conversation, only record what the user actually said"


@dataclass(frozen=True, slots=True)
class PreferenceDraft:
    """What the agent asks to remember, before it is checked against what the user said."""

    target_kind: str
    target_id: str
    target_label: str
    polarity: Polarity
    strength: int
    hardness: Hardness
    evidence_quote: str
    resolved_ids: tuple[str, ...] = ()
    affected_films: int = 0
    scope: Scope = "session"
    source: Source = "agent"
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class Preference:
    """One live row, as everything downstream reads it."""

    pref_id: int
    target_kind: str
    target_id: str
    target_label: str
    resolved_ids: tuple[str, ...]
    affected_films: int
    polarity: Polarity
    strength: int
    hardness: Hardness
    scope: Scope
    evidence_quote: str
    confirmed: bool
    created_at: str

    @property
    def targets(self) -> tuple[str, ...]:
        """Every corpus id this preference touches, the target itself when nothing resolved."""
        return self.resolved_ids or (self.target_id,)

    @property
    def is_hard(self) -> bool:
        """Whether this compiles to a SQL exclusion rather than a score penalty."""
        return self.hardness == "hard" and self.strength >= HARD_STRENGTH


@dataclass(frozen=True, slots=True)
class PreferenceFilter:
    """The exclusions a hard preference compiles to, which is why this is a table."""

    exclude: Mapping[str, frozenset[str]]
    require: Mapping[str, frozenset[str]]
    max_runtime: int | None = None

    def __bool__(self) -> bool:
        return bool(self.exclude or self.require or self.max_runtime)


_ACTIVE = (
    "select * from preferences where (scope = 'durable' or session_id = ?) "
    "and created_at <= coalesce(?, created_at) "
    "and (superseded_at is null or superseded_at > coalesce(?, superseded_at)) "
    "order by created_at, pref_id"
)


def ensure_session(
    db: Database, session_id: str, *, chat_provider: str = "fake", chat_model: str = "none"
) -> None:
    """Make a session row exist so session scoped preferences have something to hang on."""
    stamp = now_iso()
    with db.write() as conn:
        conn.execute(
            "insert into sessions (session_id, chat_provider, chat_model, created_at, updated_at) "
            "values (?,?,?,?,?) on conflict(session_id) do update set updated_at = excluded.updated_at",
            (session_id, chat_provider, chat_model, stamp, stamp),
        )


class PreferenceStore:
    """Append only. A contradiction supersedes, nothing is ever updated in place."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def active(self, session_id: str | None, *, as_of: datetime | None = None) -> list[Preference]:
        """Durable rows plus this session's, as they stood at as_of or as they stand now."""
        stamp = as_of.isoformat(timespec="microseconds") if as_of else None
        rows = self.db.read().execute(_ACTIVE, (session_id, stamp, stamp))
        return [_row(r) for r in rows]

    def get(self, pref_id: int) -> Preference | None:
        """One row by id, superseded or not."""
        row = (
            self.db.read()
            .execute("select * from preferences where pref_id = ?", (pref_id,))
            .fetchone()
        )
        return None if row is None else _row(row)

    def as_prompt_block(self, session_id: str | None, *, max_rows: int = 20) -> str:
        """The saved preferences block the system prompt carries, newest first."""
        rows = self.active(session_id)[-max_rows:]
        if not rows:
            return ""
        lines = ["Saved preferences (the user told you these, do not re-ask):"]
        for pref in reversed(rows):
            lines.append(
                f"- {pref.polarity}s {pref.target_label} "
                f"({pref.hardness}, strength {pref.strength}, set {pref.created_at[:10]})"
            )
        return "\n".join(lines)

    def as_filter(self, session_id: str | None) -> PreferenceFilter:
        """Hard absolute preferences only. A hard cap on a vague statement looks stupid."""
        exclude: dict[str, set[str]] = {}
        require: dict[str, set[str]] = {}
        cap: int | None = None
        for pref in self.active(session_id):
            if not pref.is_hard:
                continue
            if pref.target_kind == "runtime":
                minutes = _minutes(pref.target_id)
                cap = minutes if cap is None else min(cap, minutes)
                continue
            bucket = exclude if pref.polarity == "dislike" else require
            bucket.setdefault(pref.target_kind, set()).update(pref.targets)
        return PreferenceFilter(
            exclude={k: frozenset(v) for k, v in exclude.items()},
            require={k: frozenset(v) for k, v in require.items()},
            max_runtime=cap,
        )

    def as_penalties(self, session_id: str | None) -> dict[str, float]:
        """Signed weight per soft target, negative for a dislike, keyed kind:id."""
        out: dict[str, float] = {}
        for pref in self.active(session_id):
            if pref.is_hard:
                continue
            sign = 1.0 if pref.polarity == "like" else -1.0
            for target in pref.targets:
                out[f"{pref.target_kind}:{target}"] = sign * pref.strength * PENALTY_PER_STRENGTH
        return out

    def record(
        self,
        draft: PreferenceDraft,
        *,
        session_id: str,
        user_messages: Sequence[Message],
        evidence_message_id: int | None = None,
    ) -> Preference:
        """Write a preference, refusing a quote nobody in this conversation said."""
        if not 1 <= draft.strength <= 3:
            raise PreferenceRefused(f"strength must be 1, 2 or 3, got {draft.strength}")
        if not _quoted(draft.evidence_quote, user_messages):
            raise PreferenceRefused(QUOTE_HINT)
        owner = session_id if draft.scope == "session" else None
        stamp = now_iso()
        with self.db.write() as conn:
            row = conn.execute(
                "select pref_id from preferences where target_kind = ? and target_id = ? "
                "and coalesce(session_id, '') = ? and superseded_at is null",
                (draft.target_kind, draft.target_id, owner or ""),
            ).fetchone()
            if row is not None:
                # The live slot has to be freed before the new row can take it.
                conn.execute(
                    "update preferences set superseded_at = ? where pref_id = ?",
                    (stamp, int(row["pref_id"])),
                )
            cursor = conn.execute(
                "insert into preferences (target_kind, target_id, target_label, "
                "resolved_ids_json, affected_films, polarity, strength, hardness, scope, "
                "session_id, evidence_quote, evidence_message_id, source, confirmed, created_at) "
                "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    draft.target_kind,
                    draft.target_id,
                    draft.target_label,
                    orjson.dumps(list(draft.resolved_ids)).decode(),
                    draft.affected_films,
                    draft.polarity,
                    draft.strength,
                    draft.hardness,
                    draft.scope,
                    owner,
                    draft.evidence_quote,
                    evidence_message_id,
                    draft.source,
                    int(draft.confirmed),
                    stamp,
                ),
            )
            pref_id = int(cursor.lastrowid or 0)
            if row is not None:
                # Newest wins, and the old row keeps the audit trail rather than disappearing.
                conn.execute(
                    "update preferences set superseded_by = ? where pref_id = ?",
                    (pref_id, int(row["pref_id"])),
                )
        recorded = self.get(pref_id)
        assert recorded is not None
        return recorded

    def undo(self, pref_id: int) -> bool:
        """Supersede a row. Nothing is deleted, so as_of stays exact."""
        with self.db.write() as conn:
            cursor = conn.execute(
                "update preferences set superseded_at = ? where pref_id = ? "
                "and superseded_at is null",
                (now_iso(), pref_id),
            )
        return bool(cursor.rowcount)


def _row(row: sqlite3.Row) -> Preference:
    return Preference(
        pref_id=int(row["pref_id"]),
        target_kind=str(row["target_kind"]),
        target_id=str(row["target_id"]),
        target_label=str(row["target_label"]),
        resolved_ids=tuple(orjson.loads(str(row["resolved_ids_json"]))),
        affected_films=int(row["affected_films"]),
        polarity=cast("Polarity", row["polarity"]),
        strength=int(row["strength"]),
        hardness=cast("Hardness", row["hardness"]),
        scope=cast("Scope", row["scope"]),
        evidence_quote=str(row["evidence_quote"]),
        confirmed=bool(row["confirmed"]),
        created_at=str(row["created_at"]),
    )


def _quoted(quote: str, messages: Sequence[Message]) -> bool:
    """A cheap string comparison that kills the whole class of invented memories."""
    needle = _flatten(quote)
    if not needle:
        return False
    return any(needle in _flatten(m.content) for m in messages if m.role == "user")


def _flatten(text: str) -> str:
    return " ".join(text.split()).casefold()


def _minutes(target_id: str) -> int:
    digits = "".join(c for c in target_id if c.isdigit())
    if not digits:
        raise PreferenceRefused(f"runtime preference {target_id!r} names no number of minutes")
    return int(digits)
