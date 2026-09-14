"""Which device local models run on, and the one gate that keeps them off each other."""

from __future__ import annotations

import anyio
import torch

from palate.errors import ConfigError

_limiter: anyio.CapacityLimiter | None = None


def resolve_device(preferred: str | None = None) -> str:
    """mps when torch has it, then cuda, otherwise cpu."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def mps_limiter() -> anyio.CapacityLimiter:
    """One shared gate, so a 300M embedder and a 568M reranker never share the GPU."""
    global _limiter
    if _limiter is None:
        _limiter = anyio.CapacityLimiter(1)
    return _limiter


# A wrong reranker score is a plausible ranking with no error, so this is refused, not warned.
HALF_PRECISION = frozenset({"float16", "fp16", "half", "bfloat16"})


def check_precision(dtype: str, device: str) -> str:
    """The dtype back, unless it is half precision on mps, which returns subtly wrong scores."""
    if device == "mps" and dtype in HALF_PRECISION:
        raise ConfigError(f"{dtype} on mps gives subtly wrong cross-encoder scores, use float32")
    return dtype
