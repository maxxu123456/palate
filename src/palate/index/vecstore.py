"""The vec0 table for one fingerprint, and the pre-filter path chosen by set size."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import orjson

from palate.db.sqlvec import serialize_f32
from palate.providers.fingerprint import unpack_f32

type PrefilterPath = Literal["allow_json", "exclude_json", "metadata_overfetch"]

# Ten metadata columns plus one partition key, under the vec0 cap of 16.
METADATA_COLUMNS = (
    "year",
    "runtime",
    "vote_count",
    "original_language",
    "in_corpus",
    "is_watched",
    "is_adult",
    "is_animation",
    "is_documentary",
    "has_overview",
)

# Only these operators exist on a vec0 metadata column.
_OPERATORS = {"eq": "=", "ne": "!=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}


@dataclass(frozen=True, slots=True)
class VecRow:
    """One film as vec0 stores it: the vector plus every single valued filter."""

    film_id: int
    embedding: Sequence[float]
    decade: int = -1
    year: int = -1
    runtime: int = -1
    vote_count: int = 0
    original_language: str = ""
    in_corpus: int = 1
    is_watched: int = 0
    is_adult: int = 0
    is_animation: int = 0
    is_documentary: int = 0
    has_overview: int = 0


@dataclass(frozen=True, slots=True)
class VecHit:
    """One nearest neighbour, distance first because that is what vec0 returns."""

    film_id: int
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, which is what every downstream channel wants."""
        return 1.0 - self.distance


@dataclass(frozen=True, slots=True)
class PrefilterPlan:
    """How a constrained KNN will actually be run."""

    path: PrefilterPath
    overfetch: float = 1.0


def choose_prefilter(
    n_allowed: int,
    n_corpus: int,
    *,
    id_cap: int = 5000,
    overfetch_floor: float = 2.0,
    overfetch_cap: float = 6.0,
) -> PrefilterPlan:
    """Pick the cheapest real pre-filter for this allow set."""
    if n_allowed <= id_cap:
        return PrefilterPlan("allow_json")
    if n_corpus - n_allowed <= id_cap:
        return PrefilterPlan("exclude_json")
    # Neither list fits, so push the metadata predicates down and over-fetch the rest.
    pass_rate = n_allowed / n_corpus if n_corpus else 1.0
    factor = 1.0 / pass_rate if pass_rate else overfetch_cap
    return PrefilterPlan("metadata_overfetch", min(max(factor, overfetch_floor), overfetch_cap))


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Predicates vec0 can evaluate itself, as (column, operator, value) triples."""

    clauses: tuple[tuple[str, str, Any], ...] = field(default=())

    @classmethod
    def of(cls, **equals: Any) -> MetadataFilter:
        """The common case: a handful of equality predicates."""
        return cls(tuple((k, "eq", v) for k, v in equals.items() if v is not None))

    def sql(self) -> tuple[str, list[Any]]:
        """The where fragment and its parameters."""
        parts: list[str] = []
        params: list[Any] = []
        for column, operator, value in self.clauses:
            if column not in METADATA_COLUMNS and column != "decade":
                raise ValueError(f"{column} is not a vec0 metadata column")
            parts.append(f"{column} {_OPERATORS[operator]} ?")
            params.append(value)
        return " ".join(f"and {p}" for p in parts), params


class VecStore:
    """One runtime vec0 table. The dimension is fixed when it is created."""

    def __init__(self, conn: sqlite3.Connection, *, table: str, dim: int) -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError(f"bad vec table name {table!r}")
        self.conn = conn
        self.table = table
        self.dim = dim

    def create(self) -> None:
        """Create the table at this fingerprint's dimension, if it is not there."""
        columns = ",\n  ".join(
            f"{name} {'text' if name == 'original_language' else 'integer'}"
            for name in METADATA_COLUMNS
        )
        self.conn.execute(
            f"create virtual table if not exists {self.table} using vec0(\n"
            "  film_id integer primary key,\n"
            "  decade integer partition key,\n"
            f"  embedding float[{self.dim}] distance_metric=cosine,\n"
            f"  {columns}\n)"
        )

    def drop(self) -> None:
        """Remove the table. The index row is what says whether that was allowed."""
        self.conn.execute(f"drop table if exists {self.table}")

    def exists(self) -> bool:
        """Whether the runtime table is actually there."""
        found = self.conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (self.table,)
        ).fetchone()
        return found is not None

    def count(self) -> int:
        """How many vectors are stored."""
        return int(self.conn.execute(f"select count(*) from {self.table}").fetchone()[0])

    def ids(self) -> set[int]:
        """Every film id in the table, for the orphan check."""
        return {int(r[0]) for r in self.conn.execute(f"select film_id from {self.table}")}

    def vectors(self, film_ids: Sequence[int]) -> dict[int, tuple[float, ...]]:
        """Stored vectors for these films, which is what anything fitted in this space needs."""
        if not film_ids:
            return {}
        rows = self.conn.execute(
            f"select film_id, embedding from {self.table} "
            "where film_id in (select value from json_each(?))",
            (orjson.dumps(list(film_ids)).decode(),),
        )
        return {int(r["film_id"]): unpack_f32(bytes(r["embedding"])) for r in rows}

    def upsert(self, rows: Sequence[VecRow]) -> None:
        """Replace these films' vectors. vec0 has no upsert, so a delete comes first."""
        if not rows:
            return
        names = ("film_id", "decade", "embedding", *METADATA_COLUMNS)
        placeholders = ",".join("?" * len(names))
        self.conn.executemany(
            f"delete from {self.table} where film_id = ?", [(r.film_id,) for r in rows]
        )
        self.conn.executemany(
            f"insert into {self.table} ({', '.join(names)}) values ({placeholders})",
            [
                (
                    row.film_id,
                    row.decade,
                    serialize_f32(row.embedding),
                    *(getattr(row, name) for name in METADATA_COLUMNS),
                )
                for row in rows
            ],
        )

    def knn(
        self,
        vector: Sequence[float],
        *,
        k: int,
        allow: Sequence[int] | None = None,
        exclude: Sequence[int] | None = None,
        where: MetadataFilter | None = None,
    ) -> tuple[VecHit, ...]:
        """Nearest neighbours. Every path goes through here, so k is never missing."""
        if k <= 0:
            return ()
        clause, params = (where or MetadataFilter()).sql()
        sql = [f"select film_id, distance from {self.table} where embedding match ? and k = ?"]
        args: list[Any] = [serialize_f32(vector), k]
        if allow is not None:
            sql.append("and film_id in (select value from json_each(?))")
            args.append(orjson.dumps(list(allow)).decode())
        if exclude is not None:
            sql.append("and film_id not in (select value from json_each(?))")
            args.append(orjson.dumps(list(exclude)).decode())
        if clause:
            sql.append(clause)
            args.extend(params)
        rows = self.conn.execute(" ".join(sql), args)
        return tuple(VecHit(int(r["film_id"]), float(r["distance"])) for r in rows)
