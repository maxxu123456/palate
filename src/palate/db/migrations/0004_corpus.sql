-- The queue is the crawl's only state, so a killed worker leaves a reclaimable
-- lease behind rather than a half-finished pass that needs a recovery step.

create table corpus_members (
  tmdb_id           integer primary key references films(tmdb_id) on delete cascade,
  source            text not null check (source in ('discover','history','onehop','export')),
  eligible          integer not null default 1,
  ineligible_reason text,                  -- kept, never deleted, so counts stay queryable
  vote_floor_used   integer not null default 25,
  added_at          text not null
) strict;
create index corpus_eligible_idx on corpus_members(eligible) where eligible = 1;
create index corpus_source_idx   on corpus_members(source);

create table crawl_runs (
  run_id      text primary key,
  kind        text not null check (kind in ('discover','detail','onehop','export')),
  params_json text,
  started_at  text not null,
  finished_at text,
  status      text not null check (status in ('running','done','failed','cancelled')),
  n_ok    integer not null default 0,
  n_304   integer not null default 0,
  n_err   integer not null default 0,
  n_dead  integer not null default 0
) strict;

create table crawl_queue (
  id              integer primary key autoincrement,
  kind            text not null check (kind in ('discover','detail','onehop','export')),
  tmdb_id         integer,
  params_json     text,
  priority        integer not null default 0,
  state           text not null default 'pending'
                  check (state in ('pending','leased','done','failed','dead')),
  attempts        integer not null default 0,
  lease_until     real,
  next_attempt_at real not null default 0,
  last_error      text,
  run_id          text references crawl_runs(run_id)
) strict;
create unique index crawl_queue_key
  on crawl_queue(kind, coalesce(tmdb_id, -1), coalesce(params_json, ''));
create index crawl_queue_ready on crawl_queue(state, next_attempt_at, priority desc);
create index crawl_queue_lease on crawl_queue(state, lease_until) where state = 'leased';
