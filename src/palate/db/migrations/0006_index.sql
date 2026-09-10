-- vec_films_<index_id> is created at runtime, never here: vec0 fixes the dimension at
-- CREATE time and the dimension is not known until a fingerprint is chosen.

create table embedding_indexes (
  index_id             text primary key,   -- the fingerprint key, 16 hex
  provider             text not null,
  model_id             text not null,
  revision             text,
  dim                  integer not null,
  normalized           integer not null,
  query_prompt         text not null,
  document_prompt      text not null,
  pooling              text not null,
  doc_template_version text not null,
  table_name           text not null unique,     -- vec_films_<index_id>, created at runtime
  canary_text          text not null,
  canary_vec           blob not null,
  canary_checked_at    text,
  n_vectors            integer not null default 0,
  status               text not null default 'building'
                       check (status in ('building','ready','stale','failed')),
  created_at           text not null,
  completed_at         text
) strict;

-- At most one active index. Flipping the pointer is one UPDATE inside one transaction.
create table active_index (
  only_row integer primary key check (only_row = 1),
  index_id text not null references embedding_indexes(index_id)
) strict;

create table film_embeddings (
  index_id    text not null references embedding_indexes(index_id) on delete cascade,
  tmdb_id     integer not null references films(tmdb_id) on delete cascade,
  doc_sha     text not null,              -- an incremental refresh re-embeds only movers
  embedded_at text not null,
  primary key (index_id, tmdb_id)
) strict, without rowid;

create table embedding_cache (
  fingerprint_key text not null,
  text_sha        text not null,
  vec             blob not null,
  created_at      text not null,
  hits            integer not null default 0,
  primary key (fingerprint_key, text_sha)
) strict, without rowid;
