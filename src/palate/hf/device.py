"""Which device local models run on, and the one gate that keeps them off each other."""

from __future__ import annotations

import anyio

from palate.extras import have, require

_limiter: anyio.CapacityLimiter | None = None


def resolve_device(preferred: str | None = None) -> str:
    """mps when torch says it is there, otherwise cpu. Never imports torch to say cpu."""
    if preferred:
        return preferred
    if not have("torch"):
        return "cpu"
    torch = require("local", "torch")
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def mps_limiter() -> anyio.CapacityLimiter:
    """One shared gate, so a 300M embedder and a 568M reranker never share the GPU."""
    global _limiter
    if _limiter is None:
        _limiter = anyio.CapacityLimiter(1)
    return _limiter
