-- traces.db is a second file on purpose: it is disposable, it is what gc empties, and
-- nothing in palate.db depends on a row in here.

create table runs (
  run_id           text primary key,
  kind             text not null,          -- chat | eval | index | crawl | replay
  session_id       text,
  started_at       text not null,
  ended_at         text,
  latency_ms       real,
  status           text not null check (status in ('running','ok','error','cancelled')),
  error_type       text,
  error_message    text,
  input_text       text,
  output_text      text,
  turns            integer not null default 0,
  total_tokens_in  integer not null default 0,
  total_tokens_out integer not null default 0,
  total_cost_usd   real not null default 0.0,
  cost_complete    integer not null default 1,   -- 0 when any call priced unknown
  git_sha          text,
  config_sha       text
) strict;
create index runs_time_idx on runs(started_at desc);
create index runs_kind_idx on runs(kind, started_at desc);

create table spans (
  span_id       text primary key,
  run_id        text not null references runs(run_id) on delete cascade,
  parent_id     text,
  name          text not null,
  kind          text not null check (kind in
                  ('run','llm','embed','rerank','tool','retrieval','ground','http','db')),
  seq           integer not null,          -- monotonic within a run, a stable ordering
  started_at    text not null,
  ended_at      text,
  latency_ms    real,
  status        text not null check (status in ('ok','error')),
  error_type    text,
  error_message text,
  attrs_json    text                       -- gen_ai.* semconv names where they apply
) strict;
create index spans_run_idx    on spans(run_id, seq);
create index spans_parent_idx on spans(parent_id);
create index spans_kind_idx   on spans(kind, started_at desc);

create table trace_stats (
  key   text primary key,                  -- spans_dropped, bytes_written, ...
  value integer not null
) strict;
