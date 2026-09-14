"""The settings tree, its precedence order, and secret resolution by env var name."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from palate.errors import MissingSecret
from palate.paths import config_dir


class ChatSettings(BaseModel):
    model_config = {"extra": "forbid"}
    provider: Literal[
        "transformers", "ollama", "openrouter", "openai_compat", "hf_inference", "fake"
    ] = "transformers"
    model: str = "qwen2.5-3b-instruct"
    base_url: str | None = None
    device: str | None = None
    api_key_env: str = "PALATE_CHAT_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 90.0
    num_ctx: int | None = 8192
    keep_alive: str = "10m"
    hf_provider: str = "auto"
    openrouter_order: list[str] = Field(default_factory=list)
    openrouter_allow_fallbacks: bool = True
    # Turn 0 rescue for small local models that emit tool calls as text.
    content_toolcall_parse: bool = True
    force_tools_on_turn0: bool = True


class EmbedSettings(BaseModel):
    model_config = {"extra": "forbid"}
    provider: Literal[
        "ollama", "sentence_transformers", "openai_compat", "hf_inference", "fake"
    ] = "ollama"
    model: str = "embeddinggemma"
    base_url: str | None = None
    api_key_env: str = "PALATE_EMBED_API_KEY"
    device: str | None = None
    batch_size: int = 64
    truncate_dim: int | None = None
    canary_max_age_hours: int = 24
    canary_tolerance: float = 0.999


class RerankSettings(BaseModel):
    model_config = {"extra": "forbid"}
    provider: Literal["cross_encoder", "llm", "identity"] = "cross_encoder"
    model: str = "minilm"
    depth: int = 100
    top_k: int = 30
    llm_window: int = 20
    llm_stride: int = 10


class RetrievalSettings(BaseModel):
    model_config = {"extra": "forbid"}
    pool_max: int = 1200
    dense_per_mode: int = 200
    dense_query: int = 400
    bm25: int = 300
    people: int = 300
    keyword: int = 300
    popular: int = 200
    rrf_k: int = 60
    prefilter_id_cap: int = 5000
    overfetch_floor: float = 2.0
    overfetch_cap: float = 6.0
    diversity: Literal["auto", "on", "off"] = "auto"
    max_per_director: int = 2
    max_per_decade: int = 3
    max_per_collection: int = 1
    max_known_directors: int = 4


class AgentSettings(BaseModel):
    model_config = {"extra": "forbid"}
    max_turns: int = 8
    max_tool_calls: int = 16
    max_parallel_calls: int = 4
    max_prompt_tokens: int = 24_000
    max_completion_tokens: int = 4_000
    max_cost_usd: float = 0.10
    max_wall_s: float = 90.0
    max_consecutive_tool_errors: int = 3
    max_repeat: int = 3
    strict_grounding: bool = True
    nli_threshold: float = 0.5


class TMDBSettings(BaseModel):
    model_config = {"extra": "forbid"}
    token_env: str = "TMDB_READ_TOKEN"
    rate_per_s: float = 20.0
    burst: int = 20
    concurrency: int = 8
    lease_s: int = 120
    language: str = "en-US"
    corpus_target: int = 40_000
    discover_since: int = 1920
    # Zero vote films are never eligible, and dropping them keeps the windows wide.
    discover_vote_floor: int = 1


class TraceSettings(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = True
    payloads: Literal["off", "hashed", "full"] = "hashed"
    queue_max: int = 10_000
    flush_ms: int = 200
    batch: int = 64
    retain_days: int = 30


class EvalSettings(BaseModel):
    model_config = {"extra": "forbid"}
    seed: int = 0
    cuts: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
    catalogue_day_threshold: float = 0.02
    min_reliable: int = 300
    min_test_per_fold: int = 25
    bootstrap_resamples: int = 1000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PALATE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
    )
    home: Path = Field(default_factory=lambda: Path.home() / ".palate")
    offline: bool = False
    chat: ChatSettings = ChatSettings()
    embed: EmbedSettings = EmbedSettings()
    rerank: RerankSettings = RerankSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()
    tmdb: TMDBSettings = TMDBSettings()
    trace: TraceSettings = TraceSettings()
    eval: EvalSettings = EvalSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Resolved per call, not at import, so XDG_CONFIG_HOME and cwd still matter.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=[config_dir() / "palate.toml", Path("palate.toml")],
                deep_merge=True,
            ),
        )


def load_settings(**overrides: Any) -> Settings:
    """Build Settings, applying CLI overrides at the highest precedence."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)


def resolve_secret(var: str) -> SecretStr | None:
    """Read a secret by env var name. Never stored in Settings."""
    value = os.environ.get(var)
    return SecretStr(value) if value else None


def require_secret(var: str) -> SecretStr:
    """Same, but a missing variable is an error naming the variable."""
    secret = resolve_secret(var)
    if secret is None:
        raise MissingSecret(var)
    return secret


def _mask(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key.endswith("_env") and isinstance(value, str):
                status = "set" if os.environ.get(value) else "missing"
                out[key] = {"env": value, "status": status}
            else:
                out[key] = _mask(value)
        return out
    if isinstance(node, list | tuple):
        return [_mask(v) for v in node]
    if isinstance(node, Path):
        return str(node)
    return node


def masked_dump(s: Settings) -> dict[str, object]:
    """Every *_env key becomes {'env': NAME, 'status': 'set' | 'missing'}."""
    return cast("dict[str, object]", _mask(s.model_dump()))
