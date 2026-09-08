"""Pydantic models for the parts of a TMDB payload that are actually stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# TMDB adds fields constantly, so unknown keys are ignored rather than fatal.
_LENIENT = ConfigDict(extra="ignore")


def year_of(release_date: str | None) -> int | None:
    """Year from a TMDB date, which is an empty string for unreleased films."""
    if not release_date or len(release_date) < 4 or not release_date[:4].isdigit():
        return None
    return int(release_date[:4])


class GenreRef(BaseModel):
    model_config = _LENIENT
    id: int
    name: str


class KeywordRef(BaseModel):
    model_config = _LENIENT
    id: int
    name: str


class CountryRef(BaseModel):
    model_config = _LENIENT
    iso_3166_1: str
    name: str = ""


class LanguageRef(BaseModel):
    model_config = _LENIENT
    iso_639_1: str
    name: str = ""
    english_name: str = ""


class CompanyRef(BaseModel):
    model_config = _LENIENT
    id: int
    name: str
    origin_country: str | None = None


class CollectionRef(BaseModel):
    model_config = _LENIENT
    id: int
    name: str


class CastCredit(BaseModel):
    model_config = _LENIENT
    credit_id: str
    id: int
    name: str
    character: str | None = None
    order: int = 0
    gender: int | None = None
    known_for_department: str | None = None
    popularity: float | None = None


class CrewCredit(BaseModel):
    model_config = _LENIENT
    credit_id: str
    id: int
    name: str
    job: str | None = None
    department: str | None = None
    gender: int | None = None
    known_for_department: str | None = None
    popularity: float | None = None


class CreditsBlock(BaseModel):
    model_config = _LENIENT
    cast: list[CastCredit] = Field(default_factory=list)
    crew: list[CrewCredit] = Field(default_factory=list)


class KeywordsBlock(BaseModel):
    model_config = _LENIENT
    keywords: list[KeywordRef] = Field(default_factory=list)


class ExternalIds(BaseModel):
    model_config = _LENIENT
    imdb_id: str | None = None


class ReleaseDate(BaseModel):
    model_config = _LENIENT
    certification: str | None = None
    release_date: str | None = None
    type: int | None = None


class ReleaseDatesByCountry(BaseModel):
    model_config = _LENIENT
    iso_3166_1: str
    release_dates: list[ReleaseDate] = Field(default_factory=list)


class ReleaseDatesBlock(BaseModel):
    model_config = _LENIENT
    results: list[ReleaseDatesByCountry] = Field(default_factory=list)


class MovieDetail(BaseModel):
    """A /movie/{id} payload with append_to_response, keeping only stored fields."""

    model_config = _LENIENT
    id: int
    imdb_id: str | None = None
    title: str
    original_title: str | None = None
    release_date: str | None = None
    runtime: int | None = None
    original_language: str | None = None
    overview: str | None = None
    tagline: str | None = None
    popularity: float | None = None
    vote_average: float | None = None
    vote_count: int = 0
    budget: int | None = None
    revenue: int | None = None
    adult: bool = False
    status: str | None = None
    poster_path: str | None = None
    belongs_to_collection: CollectionRef | None = None
    genres: list[GenreRef] = Field(default_factory=list)
    production_countries: list[CountryRef] = Field(default_factory=list)
    spoken_languages: list[LanguageRef] = Field(default_factory=list)
    production_companies: list[CompanyRef] = Field(default_factory=list)
    credits: CreditsBlock = Field(default_factory=CreditsBlock)
    keywords: KeywordsBlock = Field(default_factory=KeywordsBlock)
    external_ids: ExternalIds = Field(default_factory=ExternalIds)
    release_dates: ReleaseDatesBlock = Field(default_factory=ReleaseDatesBlock)

    @property
    def year(self) -> int | None:
        """Year of the primary release date."""
        return year_of(self.release_date)

    @property
    def release_years(self) -> tuple[int, ...]:
        """Every year the film opened somewhere, which is what a reissue needs."""
        years = {
            year
            for country in self.release_dates.results
            for entry in country.release_dates
            if (year := year_of(entry.release_date)) is not None
        }
        return tuple(sorted(years))

    @property
    def primary_region(self) -> str | None:
        """First production country, which is what the vote floor keys off."""
        return self.production_countries[0].iso_3166_1 if self.production_countries else None


class MovieSummary(BaseModel):
    """One result row from discover, search, recommendations or similar."""

    model_config = _LENIENT
    id: int
    title: str = ""
    original_title: str | None = None
    release_date: str | None = None
    overview: str | None = None
    popularity: float | None = None
    vote_average: float | None = None
    vote_count: int = 0
    adult: bool = False

    @property
    def year(self) -> int | None:
        """Year of the release date carried in the summary."""
        return year_of(self.release_date)


class MoviePage(BaseModel):
    """A paged list response. total_results is what the window planner reads."""

    model_config = _LENIENT
    page: int = 1
    total_pages: int = 1
    total_results: int = 0
    results: list[MovieSummary] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DiscoverParams:
    """The /discover/movie query, as the crawl varies it."""

    sort_by: str = "popularity.desc"
    vote_count_gte: int | None = None
    primary_release_date_gte: str | None = None
    primary_release_date_lte: str | None = None
    with_original_language: str | None = None
    region: str | None = None
    include_adult: bool = False

    def as_query(self) -> dict[str, str]:
        """Query parameters in TMDB's own spelling."""
        query: dict[str, str] = {
            "sort_by": self.sort_by,
            "include_adult": "true" if self.include_adult else "false",
            "include_video": "false",
        }
        if self.vote_count_gte is not None:
            query["vote_count.gte"] = str(self.vote_count_gte)
        if self.primary_release_date_gte:
            query["primary_release_date.gte"] = self.primary_release_date_gte
        if self.primary_release_date_lte:
            query["primary_release_date.lte"] = self.primary_release_date_lte
        if self.with_original_language:
            query["with_original_language"] = self.with_original_language
        if self.region:
            query["region"] = self.region
        return query

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the queue row, so a discover window survives a restart."""
        return {
            "sort_by": self.sort_by,
            "vote_count_gte": self.vote_count_gte,
            "primary_release_date_gte": self.primary_release_date_gte,
            "primary_release_date_lte": self.primary_release_date_lte,
            "with_original_language": self.with_original_language,
            "region": self.region,
            "include_adult": self.include_adult,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoverParams:
        """Rebuild from a queue row, ignoring keys an older version did not write."""
        fields = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**fields)
