"""What counts as relevant. The cutpoints move every NDCG, so they live in one place."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from palate.taste.signals import RatedFilm

GRADED: Mapping[int, int] = {10: 3, 9: 3, 8: 2, 7: 1}

# Three stars and below. Present in the test set at relevance zero, not left unjudged.
HARD_NEGATIVE_MAX = 6


def graded_relevance(rating_half: int) -> int:
    """Five and four and a half stars are 3, four is 2, three and a half is 1, the rest are 0."""
    return GRADED.get(rating_half, 0)


def watch_relevance(_: int) -> int:
    """Any test-window watch counts as 1, which is the easier task on purpose."""
    return 1


def is_hard_negative(rating_half: int) -> bool:
    """A film the user watched and did not like, which is what separates the arms."""
    return rating_half <= HARD_NEGATIVE_MAX


def labels_of(film: RatedFilm) -> tuple[int, int]:
    """Both labels for one rated film, in the order the split stores them."""
    return graded_relevance(film.rating_half), watch_relevance(film.rating_half)


def histogram(rated: Sequence[RatedFilm]) -> dict[str, int]:
    """Rating counts by half star, printed beside the cutpoints so they can be argued with."""
    out: dict[str, int] = {}
    for film in rated:
        key = str(film.rating_half)
        out[key] = out.get(key, 0) + 1
    return {k: out[k] for k in sorted(out, key=int)}
