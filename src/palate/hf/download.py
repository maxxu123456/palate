"""Fetching a pinned snapshot, and failing loudly instead of hanging when offline."""

from __future__ import annotations

import os
from pathlib import Path

from palate.errors import MissingModel
from palate.extras import require
from palate.hf.models import ModelPin


def offline() -> bool:
    """Whether the Hub is off limits for this process."""
    return os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")


def ensure_local(pin: ModelPin, *, allow_download: bool | None = None) -> Path:
    """Path to the snapshot at the pinned sha, downloading it only when that is allowed."""
    hub = require("hf", "huggingface_hub")
    download = not offline() if allow_download is None else allow_download
    try:
        path = hub.snapshot_download(
            repo_id=pin.repo_id, revision=pin.revision, local_files_only=not download
        )
    except Exception as exc:
        # Every hub miss looks different, and none of them says what to run next.
        raise MissingModel(pin.alias, pin.repo_id, pin.revision) from exc
    return Path(path)
