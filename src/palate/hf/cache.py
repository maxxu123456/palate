"""What the local Hub cache holds, so a claimed revision can be checked offline."""

from __future__ import annotations

from dataclasses import dataclass

from palate.extras import require


@dataclass(frozen=True, slots=True)
class CachedRepo:
    """One repo in the local cache, with every revision it has on disk."""

    repo_id: str
    revisions: tuple[str, ...]
    size_bytes: int

    def holds(self, revision: str) -> bool:
        """True when this exact commit is present, not just some commit."""
        return any(r.startswith(revision) or revision.startswith(r) for r in self.revisions)


def scan() -> tuple[CachedRepo, ...]:
    """Every cached repo, largest first."""
    hub = require("hf", "huggingface_hub")
    info = hub.scan_cache_dir()
    repos = [
        CachedRepo(
            repo_id=repo.repo_id,
            revisions=tuple(rev.commit_hash for rev in repo.revisions),
            size_bytes=int(repo.size_on_disk),
        )
        for repo in info.repos
    ]
    return tuple(sorted(repos, key=lambda r: -r.size_bytes))


def holds(repo_id: str, revision: str) -> bool:
    """Whether the cache really has that commit, which is how a lying pin is caught."""
    return any(r.repo_id == repo_id and r.holds(revision) for r in scan())


def total_bytes() -> int:
    """Disk the Hub cache is using, which the doctor prints."""
    return sum(r.size_bytes for r in scan())
