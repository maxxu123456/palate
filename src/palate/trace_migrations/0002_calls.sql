-- One row per outbound call, under the span that made it. messages_z and response_z stay
-- null unless trace.payloads is full, which is what keeps the viewing history in one file.

create table llm_calls (
  span_id         text primary key references spans(span_id) on delete cascade,
  run_id          text not null,
  provider        text not null,
  request_model   text not null,
  response_model  text,
  prompt_name     text,
  prompt_version  text,
  prompt_sha      text,
  messages_z      blob,
  messages_sha    text not null,
  response_z      blob,
  response_sha    text,
  tools_json      text,
  tool_calls_json text,
  finish_reason   text,
  temperature     real,
  seed            integer,
  tokens_in       integer not null default 0,
  tokens_out      integer not null default 0,
  tokens_cached   integer not null default 0,
  tokens_exact    integer not null default 1,
  cost_usd        real,
  cost_source     text not null check (cost_source in ('provider','table','unknown')),
  cache_hit       integer not null default 0,
  attempt         integer not null default 1,
  ttft_ms         real,
  latency_ms      real
) strict;
create index llm_calls_model_idx on llm_calls(provider, request_model);
create index llm_calls_run_idx   on llm_calls(run_id);

create table embed_calls (
  span_id         text primary key references spans(span_id) on delete cascade,
  run_id          text not null,
  provider        text not null,
  model_id        text not null,
  fingerprint_key text not null,
  kind            text not null check (kind in ('query','document')),
  n_texts         integer not null,
  n_cache_hits    integer not null default 0,
  tokens_in       integer,
  cost_usd        real,
  cost_source     text not null,
  latency_ms      real
) strict;

create table rerank_calls (
  span_id      text primary key references spans(span_id) on delete cascade,
  run_id       text not null,
  model_key    text not null,
  doc_version  text not null,
  n_pairs      integer not null,
  n_cache_hits integer not null default 0,
  cold_start   integer not null default 0,
  cost_usd     real,
  cost_source  text not null,
  latency_ms   real
) strict;

create table tool_calls (
  span_id          text primary key references spans(span_id) on delete cascade,
  run_id           text not null,
  turn             integer not null,
  seq_in_turn      integer not null,
  tool_name        text not null,
  tool_call_id     text,
  args_json        text,
  args_fingerprint text not null,          -- sha256(name + canonical args), repeat detection
  args_valid       integer not null,
  validation_error text,                   -- the exact text handed back to the model
  result_json      text,
  result_sha       text,
  result_rows      integer,
  result_bytes     integer not null default 0,
  truncated        integer not null default 0,
  cache_hit        integer not null default 0,
  ok               integer not null,
  error_code       text,
  latency_ms       real
) strict;
create index tool_calls_name_idx on tool_calls(tool_name, ok);
create index tool_calls_run_idx  on tool_calls(run_id, turn, seq_in_turn);

create table http_calls (
  span_id     text primary key references spans(span_id) on delete cascade,
  run_id      text,
  host        text not null,
  method      text not null,
  path        text not null,               -- never a query string, keys live in headers
  status      integer,
  retry_after real,
  attempt     integer not null default 1,
  from_cache  integer not null default 0,
  bytes       integer,
  latency_ms  real
) strict;
create index http_calls_host_idx on http_calls(host, status);

create table grounding_claims (
  claim_id        integer primary key autoincrement,
  run_id          text not null references runs(run_id) on delete cascade,
  sentence        text not null,
  film_id         integer,
  claim_kind      text not null check (claim_kind in
                    ('entity','number','user_rating','plot','opinion')),
  method          text not null check (method in ('set','exact','span','nli','exempt')),
  supported       integer not null,
  evidence_source text,
  score           real,
  detail          text
) strict;
create index grounding_run_idx on grounding_claims(run_id, supported);
