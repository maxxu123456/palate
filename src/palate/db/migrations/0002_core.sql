create table films (
  tmdb_id           integer primary key,
  imdb_id           text,
  title             text not null,
  original_title    text,
  release_date      text,
  year              integer,
  runtime           integer,
  original_language text,
  overview          text,
  tagline           text,
  popularity        real,                 -- today's value, leaks forward
  popularity_at_crawl real,               -- frozen at crawl time
  vote_average      real,
  vote_count        integer not null default 0,
  budget            integer,
  revenue           integer,
  adult             integer not null default 0,
  status            text,
  poster_path       text,
  collection_id     integer,
  collection_name   text,
  detail_version    integer not null default 1,   -- bumped when normalize changes
  etag              text,
  fetched_at        text not null,
  decade            integer generated always as ((year / 10) * 10) virtual,
  runtime_bucket    integer generated always as (
                      case when runtime is null then -1
                           when runtime <  85 then 0
                           when runtime < 105 then 1
                           when runtime < 130 then 2
                           when runtime < 160 then 3
                           else 4 end) virtual
) strict;
create index films_year_idx    on films(year);
create index films_decade_idx  on films(decade);
create index films_lang_idx    on films(original_language);
create index films_votes_idx   on films(vote_count desc);
create index films_imdb_idx    on films(imdb_id) where imdb_id is not null;
create index films_coll_idx    on films(collection_id) where collection_id is not null;

create table people (
  person_id            integer primary key,
  name                 text not null,
  gender               integer,
  known_for_department text,
  popularity           real
) strict;
create index people_name_idx on people(name collate nocase);

create table credits (
  credit_id   text primary key,           -- TMDB credit_id, stable and unique
  tmdb_id     integer not null references films(tmdb_id) on delete cascade,
  person_id   integer not null references people(person_id),
  credit_kind text not null check (credit_kind in ('cast','crew')),
  department  text,
  job         text,
  character   text,
  ord         integer not null default 0
) strict;
create index credits_film_idx   on credits(tmdb_id, credit_kind, ord);
create index credits_person_idx on credits(person_id, credit_kind);
-- EVERY director, not a denormalized first-credited id. Co-directed films are common
-- and dropping the second director understates the director baseline.
create index credits_dir_idx    on credits(person_id, tmdb_id) where job = 'Director';
create index credits_dir_film   on credits(tmdb_id) where job = 'Director';

create table genres   (genre_id   integer primary key, name text not null unique) strict;
create table keywords (keyword_id integer primary key, name text not null unique) strict;
create table countries(iso_3166_1 text    primary key, name text not null) strict;
create table languages(iso_639_1  text    primary key, name text not null) strict;
create table companies(company_id integer primary key, name text not null,
                       origin_country text) strict;

create table film_genres (
  tmdb_id  integer not null references films(tmdb_id) on delete cascade,
  genre_id integer not null references genres(genre_id),
  primary key (tmdb_id, genre_id)
) strict, without rowid;
create index film_genres_rev on film_genres(genre_id, tmdb_id);

create table film_keywords (
  tmdb_id    integer not null references films(tmdb_id) on delete cascade,
  keyword_id integer not null references keywords(keyword_id),
  primary key (tmdb_id, keyword_id)
) strict, without rowid;
create index film_keywords_rev on film_keywords(keyword_id, tmdb_id);

create table film_countries (
  tmdb_id    integer not null references films(tmdb_id) on delete cascade,
  iso_3166_1 text not null references countries(iso_3166_1),
  primary key (tmdb_id, iso_3166_1)
) strict, without rowid;
create index film_countries_rev on film_countries(iso_3166_1, tmdb_id);

create table film_languages (
  tmdb_id   integer not null references films(tmdb_id) on delete cascade,
  iso_639_1 text not null references languages(iso_639_1),
  primary key (tmdb_id, iso_639_1)
) strict, without rowid;
create index film_languages_rev on film_languages(iso_639_1, tmdb_id);

create table film_companies (
  tmdb_id    integer not null references films(tmdb_id) on delete cascade,
  company_id integer not null references companies(company_id),
  primary key (tmdb_id, company_id)
) strict, without rowid;

-- Denormalised per-film facts, rebuilt rather than hand edited. n_directors exists so a
-- query can tell a solo credit from a co-direction without a join.
create table film_stats (
  tmdb_id        integer primary key references films(tmdb_id) on delete cascade,
  n_directors    integer not null default 0,
  n_cast         integer not null default 0,
  n_keywords     integer not null default 0,
  log_vote_count real,
  has_overview   integer not null default 0,
  is_animation   integer not null default 0,
  is_documentary integer not null default 0,
  primary_region text,                    -- first production country, for the vote floor
  computed_at    text not null
) strict;
