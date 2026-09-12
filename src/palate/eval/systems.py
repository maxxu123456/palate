"""Every arm in the table, in one registry, with both ablation directions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from palate.eval.baselines import BASELINES
from palate.hashing import short_hash
from palate.retrieval.candidates import Channel

type Fusion = Literal["rrf", "weighted", "prior"]
type RerankerName = Literal["none", "cross_encoder", "llm"]
type Diversity = Literal["auto", "on", "off"]

ALL_CHANNELS: frozenset[Channel] = frozenset(Channel)
DENSE_CHANNELS: frozenset[Channel] = frozenset({Channel.DENSE_MODE, Channel.DENSE_QUERY})


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """One arm. Two arms differing anywhere here are two rows, never one."""

    name: str
    channels: frozenset[Channel] = ALL_CHANNELS
    use_modes: bool = True
    n_modes: int | None = None
    use_ridge: bool = True
    use_repulsion: bool = True
    use_people_prior: bool = True
    use_keyword_prior: bool = True
    use_metadata_prior: bool = True
    use_exposure_features: bool = True
    fusion: Fusion = "weighted"
    reranker: RerankerName = "cross_encoder"
    reranker_model: str | None = "minilm"
    rerank_depth: int = 100
    diversity: Diversity = "auto"
    novelty_cap: bool = True
    confidence_shrinkage: bool = True
    include_credits_in_doc: bool = True
    history_cap: int | None = None
    alpha: float = 0.6

    def to_row(self) -> dict[str, Any]:
        """The json shape hashed into config_sha and stored on the run."""
        return {
            "name": self.name,
            "channels": sorted(c.value for c in self.channels),
            "use_modes": self.use_modes,
            "n_modes": self.n_modes,
            "use_ridge": self.use_ridge,
            "use_repulsion": self.use_repulsion,
            "use_people_prior": self.use_people_prior,
            "use_keyword_prior": self.use_keyword_prior,
            "use_metadata_prior": self.use_metadata_prior,
            "use_exposure_features": self.use_exposure_features,
            "fusion": self.fusion,
            "reranker": self.reranker,
            "reranker_model": self.reranker_model,
            "rerank_depth": self.rerank_depth,
            "diversity": self.diversity,
            "novelty_cap": self.novelty_cap,
            "confidence_shrinkage": self.confidence_shrinkage,
            "include_credits_in_doc": self.include_credits_in_doc,
            "history_cap": self.history_cap,
            "alpha": self.alpha,
        }

    @property
    def config_sha(self) -> str:
        """Identity of the arm, so a repeated configuration is detected and skipped."""
        return short_hash(self.to_row())

    @property
    def is_baseline(self) -> bool:
        return self.name in BASELINES


def baseline(name: str) -> SystemConfig:
    """A baseline carries no retrieval knobs, only its name and the seed it is run under."""
    return SystemConfig(
        name=name,
        channels=frozenset(),
        use_modes=False,
        use_ridge=False,
        use_repulsion=False,
        use_people_prior=False,
        use_keyword_prior=False,
        use_metadata_prior=False,
        use_exposure_features=False,
        fusion="prior",
        reranker="none",
        reranker_model=None,
        diversity="off",
        novelty_cap=False,
        confidence_shrinkage=False,
    )


# The reranker default stays cross_encoder because that is where this is going. The full arm
# names what it actually runs today, so it does not collide with the reranker arms below.
FULL = SystemConfig(
    name="full",
    channels=ALL_CHANNELS,
    fusion="weighted",
    reranker="none",
    reranker_model=None,
)

# The naive multi-mode build: similarity and nothing else, which is where the ladder starts.
DENSE_ONLY = SystemConfig(
    name="dense_only",
    channels=DENSE_CHANNELS,
    use_ridge=False,
    use_repulsion=False,
    use_people_prior=False,
    use_keyword_prior=False,
    use_metadata_prior=False,
    use_exposure_features=False,
    fusion="prior",
    reranker="none",
    reranker_model=None,
    diversity="off",
    novelty_cap=False,
)

_WITH_BM25 = replace(DENSE_ONLY, name="+bm25", channels=DENSE_CHANNELS | {Channel.BM25})

# Adding one component at a time overstates whichever is added first, because the components
# are redundant. The leave-one-out ladder below is the other half of the same question.
ADD_IN: tuple[SystemConfig, ...] = (
    DENSE_ONLY,
    _WITH_BM25,
    replace(_WITH_BM25, name="+repulsion", use_repulsion=True),
    replace(_WITH_BM25, name="+ridge", use_ridge=True),
    replace(_WITH_BM25, name="+ridge +repulsion", use_ridge=True, use_repulsion=True),
    replace(
        _WITH_BM25,
        name="+metadata_priors",
        use_ridge=True,
        use_repulsion=True,
        use_metadata_prior=True,
    ),
    replace(
        _WITH_BM25,
        name="+exposure_features",
        use_ridge=True,
        use_repulsion=True,
        use_metadata_prior=True,
        use_exposure_features=True,
    ),
    replace(FULL, name="+people_priors", fusion="prior", diversity="off", novelty_cap=False),
)

LEAVE_OUT: tuple[SystemConfig, ...] = (
    replace(FULL, name="full - people_priors", use_people_prior=False),
    replace(FULL, name="full - keyword_priors", use_keyword_prior=False),
    replace(FULL, name="full - ridge", use_ridge=False),
    replace(FULL, name="full - repulsion", use_repulsion=False),
    replace(FULL, name="full - metadata_priors", use_metadata_prior=False),
    replace(FULL, name="full - exposure_features", use_exposure_features=False),
    replace(FULL, name="full - bm25", channels=ALL_CHANNELS - {Channel.BM25}),
    replace(FULL, name="full - modes", use_modes=False),
)

KNOBS: tuple[SystemConfig, ...] = (
    replace(FULL, name="no_diversity", diversity="off"),
    replace(FULL, name="no_novelty_cap", novelty_cap=False),
    replace(FULL, name="no_confidence_shrinkage", confidence_shrinkage=False),
    replace(FULL, name="fusion=rrf", fusion="rrf"),
    replace(FULL, name="fusion=prior", fusion="prior"),
    *(replace(FULL, name=f"rerank_depth={d}", rerank_depth=d) for d in (50, 200)),
    *(replace(FULL, name=f"alpha={a}", alpha=a) for a in (0.4, 0.8, 1.0)),
    *(replace(FULL, name=f"history_cap={c}", history_cap=c) for c in (100, 300, 1000)),
)

# Three rerankers and the identity control, which is `full` itself. The reranker_model is the
# key the eval context supplies, so an arm runs on the machine that has that checkpoint.
RERANK: tuple[SystemConfig, ...] = (
    replace(FULL, name="+cross_encoder(minilm)", reranker="cross_encoder", reranker_model="minilm"),
    replace(FULL, name="+cross_encoder(bge-m3)", reranker="cross_encoder", reranker_model="bge-m3"),
    replace(FULL, name="+llm_rerank", reranker="llm", reranker_model="listwise"),
)

BLOCKED: tuple[SystemConfig, ...] = (
    replace(FULL, name="doc_no_credits", include_credits_in_doc=False),
)

BASELINE_ARMS: tuple[SystemConfig, ...] = tuple(baseline(name) for name in BASELINES)

ABLATIONS: tuple[SystemConfig, ...] = (
    *BASELINE_ARMS,
    *ADD_IN,
    FULL,
    *LEAVE_OUT,
    *KNOBS,
    *RERANK,
    *BLOCKED,
)

BY_NAME: Mapping[str, SystemConfig] = {cfg.name: cfg for cfg in ABLATIONS}

# The subset a smoke run covers: the bar to beat, the strawman, the ladder ends and full.
SMOKE: tuple[str, ...] = (
    "popularity",
    "director_affinity",
    "single_centroid_dense",
    "dense_only",
    "full",
)


def by_name(name: str) -> SystemConfig:
    """One arm by name, or a KeyError naming what the registry does hold."""
    if name not in BY_NAME:
        raise KeyError(f"unknown system {name!r}, known: {', '.join(sorted(BY_NAME))}")
    return BY_NAME[name]


def resolve(names: Sequence[str] | None) -> tuple[SystemConfig, ...]:
    """Arms by name, or every arm when nothing was asked for."""
    if not names:
        return ABLATIONS
    return tuple(by_name(n) for n in names)


def rerank_key(cfg: SystemConfig) -> str | None:
    """Which reranker this arm asks the eval context for, or None when it reranks nothing."""
    return None if cfg.reranker == "none" else (cfg.reranker_model or cfg.reranker)


def blocked_reason(cfg: SystemConfig, *, rerankers: Collection[str] = ()) -> str | None:
    """Why an arm cannot run here, which the report prints instead of a number."""
    key = rerank_key(cfg)
    if key is not None and key not in rerankers:
        return f"needs a {key} reranker in the eval context"
    if not cfg.include_credits_in_doc:
        return "needs a second index rendered without credits"
    return None
