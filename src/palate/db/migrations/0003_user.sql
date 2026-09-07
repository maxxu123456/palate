create table letterboxd_imports (
  import_id    integer primary key autoincrement,
  imported_at  text not null,
  zip_sha256   text not null,
  n_ratings    integer not null,
  n_diary      integer not null,
  n_watched    integer not null,
  n_watchlist  integer not null,
  n_reviews    integer not null,
  n_resolved   integer not null,
  n_unresolved integer not null
) strict;

create table user_films (
  tmdb_id        integer primary key references films(tmdb_id) on delete cascade,
  rating_half    integer check (rating_half between 1 and 10),  -- never a float
  watched_date   text,                    -- diary Watched Date, the real watch date
  logged_date    text,                    -- diary or ratings entry date
  date_source    text not null check (date_source in ('diary','ratings','none')),
  date_reliable  integer not null default 1,   -- 0 when on a detected catalogue day
  is_rewatch     integer not null default 0,
  rewatch_count  integer not null default 0,
  in_watchlist   integer not null default 0,
  watchlist_added_on text,
  liked          integer not null default 0,
  review_text    text,
  review_chars   integer not null default 0,
  review_leaks_identity integer not null default 0,
  letterboxd_uri text,
  import_id      integer references letterboxd_imports(import_id)
) strict;
create index user_films_watched_idx  on user_films(watched_date) where date_reliable = 1;
create index user_films_rating_idx   on user_films(rating_half);
create index user_films_reliable_idx on user_films(date_reliable);

create table title_resolutions (
  letterboxd_uri  text primary key,
  title           text not null,
  year            integer,
  tmdb_id         integer references films(tmdb_id),
  method          text not null check (method in ('uri','exact','fuzzy','manual','failed')),
  confidence      real not null,
  candidates_json text,                    -- top 5 (id, title, year, score)
  needs_review    integer not null default 0,
  resolved_at     text not null
) strict;
create index title_resolutions_review on title_resolutions(needs_review) where needs_review = 1;

create table unmatched_export_row (
  row_id     integer primary key autoincrement,
  source_csv text not null,
  title      text,
  year       integer,
  uri        text,
  rating     real,
  reason     text not null,
  seen_at    text not null
) strict;
