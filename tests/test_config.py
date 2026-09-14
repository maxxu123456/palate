"""Precedence, secret handling and the refusal to ignore a typo."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from palate import clock
from palate.config import Settings, load_settings, masked_dump, require_secret, resolve_secret
from palate.errors import MissingExtra, MissingSecret
from palate.extras import have, require
from palate.hashing import canonical_json, request_sha, sha256_file, sha256_hex, short_hash
from palate.ids import new_run_id, new_session_id, new_span_id
from palate.paths import Paths, resolve


def test_defaults_need_no_files() -> None:
    s = Settings()
    assert s.chat.provider == "transformers"
    assert s.chat.model == "qwen2.5-3b-instruct"
    assert s.embed.provider == "sentence_transformers"
    assert s.embed.model == "embeddinggemma"
    assert s.trace.payloads == "hashed"
    assert s.agent.max_turns == 8


def test_user_config_is_read(config_file: Path) -> None:
    config_file.write_text('[chat]\nprovider = "fake"\nmodel = "qwen2.5-7b-instruct"\n')
    s = Settings()
    assert s.chat.provider == "fake"
    assert s.chat.model == "qwen2.5-7b-instruct"
    # Untouched fields keep their defaults rather than vanishing with the table.
    assert s.chat.temperature == 0.2


def test_local_config_beats_user_config(config_file: Path, tmp_path: Path) -> None:
    config_file.write_text('[chat]\nmodel = "from-user"\ntemperature = 0.9\n')
    (tmp_path / "palate.toml").write_text('[chat]\nmodel = "from-cwd"\n')
    s = Settings()
    assert s.chat.model == "from-cwd"
    assert s.chat.temperature == 0.9


def test_env_beats_config_file(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file.write_text('[chat]\nmodel = "from-file"\n')
    monkeypatch.setenv("PALATE_CHAT__MODEL", "from-env")
    assert Settings().chat.model == "from-env"


def test_dotenv_beats_config_file(config_file: Path, tmp_path: Path) -> None:
    config_file.write_text('[chat]\nmodel = "from-file"\n')
    (tmp_path / ".env").write_text("PALATE_CHAT__MODEL=from-dotenv\n")
    assert Settings().chat.model == "from-dotenv"


def test_cli_override_beats_everything(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file.write_text('[chat]\nmodel = "from-file"\n')
    monkeypatch.setenv("PALATE_CHAT__MODEL", "from-env")
    s = load_settings(chat={"model": "from-flag"})
    assert s.chat.model == "from-flag"


def test_none_overrides_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PALATE_CHAT__MODEL", "from-env")
    assert load_settings(chat=None).chat.model == "from-env"


def test_unknown_top_level_key_is_rejected(config_file: Path) -> None:
    config_file.write_text('chatt = "typo"\n')
    with pytest.raises(ValidationError):
        Settings()


def test_unknown_nested_key_is_rejected(config_file: Path) -> None:
    config_file.write_text("[chat]\ntemprature = 0.4\n")
    with pytest.raises(ValidationError):
        Settings()


def test_bad_literal_is_rejected(config_file: Path) -> None:
    config_file.write_text('[trace]\npayloads = "everything"\n')
    with pytest.raises(ValidationError):
        Settings()


def test_no_secret_is_a_settings_field() -> None:
    dumped = repr(masked_dump(Settings()))
    assert "sk-" not in dumped
    for field in Settings.model_fields:
        assert "key" not in field or field.endswith("_env")


def test_masked_dump_reports_env_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_READ_TOKEN", "secret-value")
    dump = masked_dump(Settings())
    tmdb = dump["tmdb"]
    assert isinstance(tmdb, dict)
    assert tmdb["token_env"] == {"env": "TMDB_READ_TOKEN", "status": "set"}
    assert "secret-value" not in repr(dump)


def test_resolve_secret_reads_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_READ_TOKEN", "sk-test")
    secret = resolve_secret("TMDB_READ_TOKEN")
    assert secret is not None
    assert secret.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(secret)


def test_missing_secret_names_the_variable() -> None:
    assert resolve_secret("PALATE_NOT_SET_ANYWHERE") is None
    with pytest.raises(MissingSecret) as exc:
        require_secret("PALATE_NOT_SET_ANYWHERE")
    assert "PALATE_NOT_SET_ANYWHERE" in str(exc.value)


def test_paths_pin_hf_home(offline_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    paths = resolve(offline_env)
    assert paths.db == offline_env / "palate.db"
    assert paths.traces_db == offline_env / "traces.db"
    assert paths.hf_home.is_dir()
    assert os.environ["HF_HOME"] == str(paths.hf_home)


def test_paths_are_created_once(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "root"
    first = Paths(root).ensure()
    second = Paths(root).ensure()
    assert first == second
    assert root.is_dir()


def test_canonical_json_ignores_key_order() -> None:
    a = {"b": 1, "a": {"y": 2, "x": [3, 4]}}
    b = {"a": {"x": [3, 4], "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert request_sha(a) == request_sha(b)


def test_sha256_file_matches_sha256_of_bytes(tmp_path: Path) -> None:
    blob = b"a" * (1 << 21) + b"tail"
    target = tmp_path / "blob.bin"
    target.write_bytes(blob)
    assert sha256_file(target) == sha256_hex(blob)


def test_short_hash_is_stable_and_short() -> None:
    key = short_hash({"model": "embeddinggemma", "dim": 768})
    assert len(key) == 16
    assert key == short_hash({"dim": 768, "model": "embeddinggemma"})


def test_frozen_clock_stops_time() -> None:
    at = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    with clock.frozen(at):
        assert clock.now() == at
        assert clock.now_iso().startswith("2026-09-07T12:00:00")
        assert clock.now_ns() == int(at.timestamp() * 1_000_000_000)
    assert clock.now() != at


def test_ids_sort_in_creation_order() -> None:
    runs = [new_run_id() for _ in range(200)]
    assert runs == sorted(runs)
    assert len(set(runs)) == len(runs)
    assert new_span_id().startswith("spn_")
    assert new_session_id().startswith("ses_")


def test_require_raises_missing_extra_with_the_uv_command() -> None:
    assert have("json")
    assert not have("palate_no_such_module")
    with pytest.raises(MissingExtra) as exc:
        require("api", "palate_no_such_module")
    assert "uv sync --extra api" in str(exc.value)
    assert exc.value.extra == "api"
