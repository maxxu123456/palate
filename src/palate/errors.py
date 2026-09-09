"""Every exception the package raises, in one place."""

from __future__ import annotations


class PalateError(Exception):
    """Base for everything this package raises."""


class ConfigError(PalateError):
    """Bad or missing configuration."""


class MissingExtra(PalateError):
    """An optional dependency group is not installed."""

    def __init__(self, extra: str, module: str) -> None:
        self.extra = extra
        self.module = module
        super().__init__(f"{module} needs the {extra!r} extra. Run: uv sync --extra {extra}")


class MissingSecret(ConfigError):
    """A secret was requested by env var name and the variable is unset."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"environment variable {env_var} is not set")


class ProviderError(PalateError):
    """A model provider failed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retryable = retryable
        super().__init__(message)


class ProviderTimeout(ProviderError):
    """The provider did not answer in time."""


class ProviderRateLimited(ProviderError):
    """The provider returned 429."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, model=model, retryable=True)


class ProviderUnavailable(ProviderError):
    """The provider is down or unreachable."""


class ProviderBadRequest(ProviderError):
    """The provider rejected the request as malformed."""


class ProviderAuthError(ProviderError):
    """The key is missing, wrong or out of quota."""


class ProviderContextOverflow(ProviderError):
    """The prompt did not fit in the model's context window."""


class ModelNotFound(ProviderError):
    """The endpoint is up but does not serve this model."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        pull_hint: str = "",
    ) -> None:
        self.pull_hint = pull_hint
        super().__init__(message, provider=provider, model=model, retryable=False)


class MissingModel(PalateError):
    """A pinned Hub model is not in the local cache and offline mode forbids fetching it."""

    def __init__(self, alias: str, repo_id: str, revision: str) -> None:
        self.alias = alias
        self.repo_id = repo_id
        self.revision = revision
        self.pull_hint = f"palate models pull {alias}"
        super().__init__(f"{repo_id}@{revision[:8]} is not cached. Run: {self.pull_hint}")


class IndexingError(PalateError):
    """Something is wrong with an embedding index."""


class EmbeddingFingerprintMismatch(IndexingError):
    """The active index was built by a different embedding setup."""

    def __init__(self, message: str, *, index: object, query: object) -> None:
        self.index = index
        self.query = query
        super().__init__(message)


class NoActiveIndex(IndexingError):
    """No embedding index is marked active."""


class IndexIncomplete(IndexingError):
    """The index is missing vectors for films it claims to cover."""

    def __init__(self, n_expected: int, n_present: int) -> None:
        self.n_expected = n_expected
        self.n_present = n_present
        super().__init__(f"index holds {n_present} of {n_expected} vectors")


class StaleArtifact(IndexingError):
    """A derived artifact was built against something that is no longer active."""

    def __init__(self, kind: str, artifact_id: str, built_for: str, active: str) -> None:
        self.kind = kind
        self.artifact_id = artifact_id
        self.built_for = built_for
        self.active = active
        super().__init__(f"{kind} {artifact_id} was built for {built_for}, active is {active}")


class StorageError(PalateError):
    """SQLite refused to cooperate."""


class MigrationChecksumMismatch(StorageError):
    """An applied migration file changed on disk."""


class SqliteExtensionUnavailable(StorageError):
    """This Python cannot load SQLite extensions, so sqlite-vec is out of reach."""


class TMDBError(PalateError):
    """TMDB returned something unusable."""


class TMDBNotFound(TMDBError):
    """TMDB has no such entity."""


class DiscoverWindowTooLarge(TMDBError):
    """A discover window is past the result ceiling and cannot be split any finer."""

    def __init__(self, start: str, end: str, total: int, cap: int) -> None:
        self.start = start
        self.end = end
        self.total = total
        self.cap = cap
        super().__init__(f"{start} to {end} holds {total} results, over the {cap} cap")


class UnresolvedTitle(PalateError):
    """An export row could not be tied to a TMDB id."""

    def __init__(
        self,
        title: str,
        year: int | None,
        candidates: list[tuple[int, str, float]] | None = None,
    ) -> None:
        self.title = title
        self.year = year
        self.candidates = candidates or []
        super().__init__(f"could not resolve {title!r} ({year})")


class EvalError(PalateError):
    """The evaluation harness cannot produce an honest number."""


class StaleSplitError(EvalError):
    """The frozen split no longer matches the data under it."""


class ThinHistoryError(EvalError):
    """Too few ratings to say anything."""


class TemporalProtocolUnavailable(EvalError):
    """Not enough date-reliable ratings for a temporal split."""

    def __init__(self, n_reliable: int, required: int) -> None:
        self.n_reliable = n_reliable
        self.required = required
        super().__init__(f"{n_reliable} date-reliable ratings, {required} required")
