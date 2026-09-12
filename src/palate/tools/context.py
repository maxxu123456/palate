"""What a tool handler is allowed to reach. No write connection is exposed."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import anyio

from palate.agent.budget import BudgetLedger
from palate.config import Settings
from palate.db.connect import Database
from palate.errors import ToolDeadlineExceeded, ToolFailure
from palate.providers.base import Message, SpanLike
from palate.retrieval.recommend import Recommender
from palate.retrieval.vocab import Vocabulary
from palate.taste.memory import PreferenceStore
from palate.taste.profile import TasteProfile

NO_DATA = "no film data yet, run: palate ingest"
NO_PROFILE = "no taste profile yet, run: palate profile build"


@dataclass(slots=True)
class ToolContext:
    """One run's view of the world. Absent pieces become precondition_failed, not a crash."""

    session_id: str
    run_id: str
    turn: int = 0
    deadline: float = math.inf
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    user_messages: Sequence[Message] = ()
    settings: Settings = field(default_factory=Settings)
    db: Database | None = None
    recommender: Recommender | None = None
    profile: TasteProfile | None = None
    prefs: PreferenceStore | None = None
    vocab: Vocabulary | None = None
    span: SpanLike | None = None

    def remaining_s(self) -> float:
        """Seconds a handler has left before the run's deadline."""
        return max(0.0, self.deadline - anyio.current_time())

    def check_deadline(self) -> None:
        """Raises ToolDeadlineExceeded, caught by dispatch and returned as a timeout error."""
        if self.remaining_s() <= 0.0:
            raise ToolDeadlineExceeded("the run ran out of wall clock time")

    def require_db(self) -> Database:
        """The read side of palate.db."""
        if self.db is None:
            raise ToolFailure("precondition_failed", NO_DATA)
        return self.db

    def require_recommender(self) -> Recommender:
        """The retrieval pipeline, which needs an index and a profile behind it."""
        if self.recommender is None:
            raise ToolFailure("precondition_failed", NO_DATA)
        return self.recommender

    def require_profile(self) -> TasteProfile:
        """The fitted taste profile."""
        if self.profile is None:
            raise ToolFailure("precondition_failed", NO_PROFILE)
        return self.profile

    def require_prefs(self) -> PreferenceStore:
        """Durable preference memory."""
        if self.prefs is None:
            raise ToolFailure("precondition_failed", NO_DATA)
        return self.prefs

    def require_vocab(self) -> Vocabulary:
        """The corpus's own names, for resolving what the user said."""
        if self.vocab is None:
            raise ToolFailure("precondition_failed", NO_DATA)
        return self.vocab
