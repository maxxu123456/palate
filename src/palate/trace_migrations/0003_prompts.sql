-- The prompt text keyed by its own hash, so the template that produced a stored response is
-- recoverable six weeks later. cost_rates is authoritative for pricing, pricing.toml is a seed.

create table prompts (
  prompt_sha text primary key,
  name       text not null,
  version    text not null,
  template   text not null,
  first_seen text not null
) strict;
create index prompts_name_version on prompts(name, version);

create table cost_rates (
  provider              text not null,
  model                 text not null,
  currency              text not null default 'USD',
  input_per_mtok        real not null,
  output_per_mtok       real not null,
  cached_input_per_mtok real,
  source                text not null,     -- pricing.toml | openrouter-models | manual
  fetched_at            text not null,
  primary key (provider, model, fetched_at)
) strict;

create view current_rates as
select provider, model, currency, input_per_mtok, output_per_mtok,
       cached_input_per_mtok, source, fetched_at
  from cost_rates r
 where fetched_at = (select max(fetched_at) from cost_rates r2
                      where r2.provider = r.provider and r2.model = r.model);
