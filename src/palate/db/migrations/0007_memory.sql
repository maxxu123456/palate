create table sessions (
  session_id    text primary key,
  title         text,
  chat_provider text not null,
  chat_model    text not null,
  created_at    text not null,
  updated_at    text not null
) strict;

create table messages (
  message_id      integer primary key autoincrement,
  session_id      text not null references sessions(session_id) on delete cascade,
  run_id          text,
  seq             integer not null,
  role            text not null check (role in ('system','user','assistant','tool')),
  channel         text not null default 'answer' check (channel in ('answer','thinking')),
  content         text not null default '',
  tool_calls_json text,
  tool_call_id    text,
  tool_name       text,
  token_estimate  integer,
  compacted       integer not null default 0,
  created_at      text not null,
  unique (session_id, seq)
) strict;
create index messages_session_idx on messages(session_id, seq desc);

create table runs_local (
  run_id          text primary key,
  session_id      text not null references sessions(session_id) on delete cascade,
  user_message_id integer references messages(message_id),
  phase           text not null,
  stop_reason     text,
  turns           integer not null default 0,
  tool_calls      integer not null default 0,
  input_tokens    integer not null default 0,
  output_tokens   integer not null default 0,
  cost_usd        real not null default 0,
  wall_ms         integer,
  grounded_ratio  real,
  answer_repaired integer not null default 0,
  compactions     integer not null default 0,
  error_code      text,
  started_at      text not null,
  ended_at        text
) strict;
create index runs_local_session_idx on runs_local(session_id, started_at desc);

-- Append only. A contradiction inserts a new row and sets superseded_at and superseded_by on
-- the old one. Nothing is ever UPDATEd in place, which is what makes as_of() exact.
create table preferences (
  pref_id             integer primary key autoincrement,
  target_kind         text not null check (target_kind in
                        ('genre','keyword','director','actor','writer','country','language',
                         'decade','runtime','film','collection','freeform')),
  target_id           text not null,      -- resolved corpus id, or a slug for freeform
  target_label        text not null,      -- what the user would recognise
  resolved_ids_json   text not null,      -- the full resolved id set, may be several keywords
  affected_films      integer not null,   -- how many corpus films the target touches
  polarity            text not null check (polarity in ('like','dislike')),
  strength            integer not null check (strength between 1 and 3),
  hardness            text not null check (hardness in ('hard','soft')),
  scope               text not null check (scope in ('session','durable')),
  session_id          text references sessions(session_id) on delete cascade,
  evidence_quote      text not null,
  evidence_message_id integer references messages(message_id),
  source              text not null check (source in ('agent','user','inferred')),
  confirmed           integer not null default 0,
  created_at          text not null,
  superseded_at       text,
  superseded_by       integer references preferences(pref_id)
) strict;
create unique index preferences_live_idx
  on preferences(target_kind, target_id, coalesce(session_id, ''))
  where superseded_at is null;
create index preferences_time_idx on preferences(created_at);

create table taste_profiles (
  profile_id       text primary key,
  built_at         text not null,
  cutoff_date      text,                  -- null for the live profile, set for eval slices
  index_id         text not null references embedding_indexes(index_id),
  doc_template_version text not null,
  split_name       text,
  fold             integer,
  tier             text not null check (tier in ('cold','thin','full')),
  n_rated          integer not null,
  n_reliable_dated integer not null,
  alpha            real not null,
  calibrator_json  text not null,         -- coefficients, mu, sigma, r2
  histogram_json   text not null,         -- rating_half to count, over the fitted slice
  direction_blob   blob,                  -- float32 ridge weights, null below the ridge floor
  direction_meta_json text,               -- lam, sigma2, loocv_r2, feature names, feature means
  leverage_diag_blob blob,                -- float32 diagonal of A inverse, see note
  fusion_weights_json text,               -- one object per condition
  params_sha       text not null,
  code_sha         text,
  stale            integer not null default 0,
  stale_reason     text
) strict;
create index taste_profiles_active on taste_profiles(cutoff_date, stale);

-- The full A inverse is roughly 2.5 MB per profile and 250 MB across an ablation sweep. Only
-- the diagonal is needed for leverage at scoring time, so only the diagonal is stored. The
-- full matrix is recomputed from the SVD factors when something wants to explain a score.
create table taste_modes (
  profile_id   text not null references taste_profiles(profile_id) on delete cascade,
  polarity     text not null check (polarity in ('like','dislike')),
  mode_id      integer not null,
  centroid     blob not null,             -- float32, L2 normalised
  mass         real not null,
  n_members    integer not null,
  mean_signal  real not null,
  coherence    real not null,
  confidence   real not null,
  exemplars_json text not null,
  label        text,                      -- generated once, cached, NEVER used in scoring
  label_model  text,
  primary key (profile_id, polarity, mode_id)
) strict;

create table taste_mode_members (
  profile_id text not null,
  polarity   text not null,
  mode_id    integer not null,
  tmdb_id    integer not null,
  cosine     real not null,
  signal     real not null,
  primary key (profile_id, polarity, mode_id, tmdb_id)
) strict, without rowid;

create table taste_affinities (
  profile_id       text not null references taste_profiles(profile_id) on delete cascade,
  kind             text not null,
  entity_id        text not null,
  name             text not null,
  n                integer not null,
  raw_sum          real not null,
  affinity         real not null,         -- shrunk rating affinity
  exposure_logodds real not null,         -- watch rate against the corpus base rate, shrunk
  support_json     text not null,
  primary key (profile_id, kind, entity_id)
) strict, without rowid;
create index taste_affinities_top on taste_affinities(profile_id, kind, affinity desc);
