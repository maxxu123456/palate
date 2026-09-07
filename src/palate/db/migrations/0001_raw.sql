-- tmdb_raw comes first so films.raw lookups never forward-reference a table
-- that does not exist yet. The runner bootstraps schema_migrations before it
-- reads anything, so this file only has to agree with it.

create table if not exists schema_migrations (
  version    integer primary key,
  name       text not null,
  checksum   text not null,
  applied_at text not null
) strict;

create table app_state (
  key        text primary key,
  value      text not null,
  updated_at text not null
) strict;

create table tmdb_raw (
  raw_id      integer primary key autoincrement,
  entity      text not null check (entity in
                ('movie','discover','search','recommendations','similar','person','export')),
  entity_id   integer,
  params_sha  text,                       -- for non-id entities such as discover windows
  fetched_at  text not null,
  etag        text,
  payload_sha text not null,
  payload_z   blob not null               -- zlib level 6, about 5x on TMDB json
) strict;
create unique index tmdb_raw_key
  on tmdb_raw(entity, coalesce(entity_id, -1), coalesce(params_sha, ''));
