-- One frozen split per name. Every arm reads the frozen assignment and never recomputes,
-- which is what makes two systems score against byte identical labels.

create table eval_split (
  split_name          text primary key,
  strategy            text not null check (strategy in
                        ('rolling_origin','single_temporal','leave_last_k')),
  spec_json           text not null,
  ratings_sha256      text not null,      -- staleness guard against a new export
  corpus_size         integer not null,
  n_reliable          integer not null,
  n_catalogue_days    integer not null,
  catalogue_days_json text not null,
  n_dropped_unreliable integer not null,
  n_dropped_rewatch    integer not null,
  n_dropped_not_in_corpus integer not null,
  test_coverage       real not null,
  coverage_by_region_json text not null,
  degraded            integer not null default 0,
  degraded_reason     text,
  created_at          text not null
) strict;

create table eval_fold (
  split_name text not null references eval_split(split_name) on delete cascade,
  fold       integer not null,
  t_start    text not null,
  t_split    text not null,
  t_end      text not null,
  n_train    integer not null,
  n_inner    integer not null,
  n_val      integer not null,
  n_test     integer not null,
  n_test_pos integer not null,            -- rel >= 2
  n_test_neg integer not null,            -- the hard negatives
  idcg10     real not null,
  underpowered integer not null default 0,
  primary key (split_name, fold)
) strict;

create table eval_assignment (
  split_name text not null,
  fold       integer not null,
  tmdb_id    integer not null,
  bucket     text not null check (bucket in ('inner','val','test','excluded')),
  rel        integer not null default 0,
  rel_watch  integer not null default 0,
  reason     text,                        -- why a film is not in this fold's test bucket
  primary key (split_name, fold, tmdb_id)
) strict, without rowid;

create table eval_run (
  run_id      text primary key,
  system      text not null,
  config_json text not null,
  config_sha  text not null,
  split_name  text not null,
  fold        integer not null,
  condition   text not null check (condition in
                ('unconditioned','query_review','query_synth','query_cold','agent')),
  profile_id  text,
  chat_model  text,
  code_sha    text not null,
  seed        integer not null,
  started_at  text not null,
  elapsed_ms  real not null,
  cost_usd    real not null default 0,
  pool_size   integer not null,
  pool_recall real not null,
  prefilter_path text,
  trace_run_id text
) strict;
create index eval_run_lookup on eval_run(system, split_name, fold, condition, config_sha);

create table eval_ranking (
  run_id        text not null references eval_run(run_id) on delete cascade,
  rank          integer not null,
  tmdb_id       integer not null,
  score         real not null,
  features_json text,                     -- kept for the top 50 only
  primary key (run_id, rank)
) strict;

create table eval_metric (
  run_id text not null references eval_run(run_id) on delete cascade,
  metric text not null,
  value  real not null,
  ci_lo  real,
  ci_hi  real,
  n      integer,
  primary key (run_id, metric)
) strict;

-- Paired bootstrap, one system against the reference it is being argued with.
create table eval_delta (
  split_name     text not null,
  system         text not null,
  reference      text not null,
  condition      text not null,
  metric         text not null,
  delta          real not null,
  ci_lo          real not null,
  ci_hi          real not null,
  folds_positive integer not null,
  folds_total    integer not null,
  primary key (split_name, system, reference, condition, metric)
) strict;

create table eval_weight_stability (
  split_name text not null,
  condition  text not null,
  feature    text not null,
  mean_beta  real not null,
  sd_beta    real not null,
  sign_agree integer not null,            -- folds where the sign matched the median
  n_active   integer not null,            -- folds where L1 left it nonzero
  primary key (split_name, condition, feature)
) strict;

create table eval_case_result (
  run_id      text not null references eval_run(run_id) on delete cascade,
  case_id     text not null,
  passed      integer not null,
  detail_json text,
  primary key (run_id, case_id)
) strict;
