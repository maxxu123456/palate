"""resolve_vocabulary: guessing at a filter target produces the wrong filter silently."""

from __future__ import annotations

from functools import partial

import anyio
from pydantic import BaseModel, ConfigDict, Field

from palate.retrieval.vocab import Kind, Vocabulary
from palate.tools.context import ToolContext
from palate.tools.registry import ToolKind, ToolSpec

DESCRIPTION = (
    "Turn free text like 'musicals' or 'Russian' into the corpus ids it actually means, with "
    "how many films each touches. Call this before record_preference and before any filter "
    "whose target you are guessing at."
)

MAX_MATCHES = 6


class ResolveVocabularyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=2, max_length=120)
    kinds: list[Kind] = Field(default_factory=list, max_length=6)


class Resolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    ids: list[str]
    label: str
    affected_films: int
    confidence: float
    alternatives: list[tuple[str, str, int]] = Field(default_factory=list)


class ResolveVocabularyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[Resolution]
    meta: dict[str, object] = Field(default_factory=dict)


def resolve(vocab: Vocabulary, args: ResolveVocabularyArgs) -> ResolveVocabularyResult:
    """Every corpus set the phrase could mean, best first, with what each would touch."""
    found = vocab.resolve(args.text, prefer=args.kinds)
    if args.kinds:
        found = [m for m in found if m.kind in args.kinds]
    matches = [
        Resolution(
            kind=m.kind,
            ids=list(m.ids),
            label=m.label,
            affected_films=m.affected_films,
            confidence=m.confidence,
            alternatives=[(entity, label, films) for entity, label, films in m.alternatives],
        )
        for m in found[:MAX_MATCHES]
    ]
    return ResolveVocabularyResult(
        matches=matches,
        meta={
            "count": len(found),
            "returned": len(matches),
            "ambiguous": len(matches) > 1,
            "asked": args.text,
        },
    )


async def handler(args: ResolveVocabularyArgs, ctx: ToolContext) -> ResolveVocabularyResult:
    """The vocabulary caches its own tables, so repeated calls cost one dictionary lookup."""
    ctx.check_deadline()
    vocab = ctx.require_vocab()
    return await anyio.to_thread.run_sync(partial(resolve, vocab, args))


SPEC = ToolSpec(
    name="resolve_vocabulary",
    description=DESCRIPTION,
    args_model=ResolveVocabularyArgs,
    result_model=ResolveVocabularyResult,
    kind=ToolKind.READ,
    handler=handler,
    examples=(ResolveVocabularyArgs(text="musicals", kinds=["genre", "keyword"]),),
    cost_hint_ms=15,
    max_result_chars=3000,
)
