-- temperature and seed are IN the key and a CHECK enforces the policy structurally.
-- Caching a sampled generation and serving it as fresh makes an eval look stable when it
-- is not, and a rule that lives only in a docstring is not a rule.

create table llm_cache (
  cache_key      text primary key,        -- sha256 over the canonical request
  provider       text not null,
  model          text not null,
  prompt_sha     text not null,
  temperature    real not null,
  seed           integer,
  response_z     blob not null,
  tokens_in      integer not null,
  tokens_out     integer not null,
  created_at     text not null,
  hits           integer not null default 0,
  check (temperature = 0.0 or seed is not null)
) strict;

-- doc_version is IN the key. Changing the rendered document changes what the cross-encoder
-- reads, so a cached score from the old text is silently wrong rather than merely stale.
create table rerank_cache (
  model_key   text not null,
  query_sha   text not null,
  doc_version text not null,
  film_id     integer not null,
  score       real not null,
  created_at  text not null,
  primary key (model_key, query_sha, doc_version, film_id)
) strict, without rowid;
