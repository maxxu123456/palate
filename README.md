# palate

A film recommender built on one person's Letterboxd history. Everything runs on
the laptop: the corpus is a SQLite file, the embeddings live in the same file,
and the chat model is whatever you point it at. The interesting part is not the
chat, it is that the ranking is fitted to one viewer's ratings instead of to a
crowd.

## Setup

```sh
uv sync
mkdir -p ~/.config/palate
cp palate.toml.example ~/.config/palate/palate.toml
cp .env.example .env
```

Pick a chat provider in `palate.toml`. Ollama is the default and needs no key:

```sh
brew install ollama && ollama serve && ollama pull qwen3:8b
```

For a hosted model instead, set `provider = "openrouter"` and put the key name in
`api_key_env`. Any OpenAI-compatible `base_url` works too (LM Studio, vLLM,
Together, Groq). Secrets never go in the config file, only the name of the
environment variable that holds them.

A TMDB v4 read token goes in `TMDB_READ_TOKEN`. Without it titles resolve only
from Letterboxd URIs that already carry a TMDB id, and there is nothing to crawl
with.

Then export your Letterboxd data (Settings, Data, Export your data), import it,
and fill the corpus:

```sh
palate ingest ~/Downloads/letterboxd-2026-09-14.zip
palate ingest review
palate tmdb crawl
palate tmdb status
```

## Configuration

Precedence, highest first: CLI flag, `PALATE_` environment variables, `.env`,
`./palate.toml`, `~/.config/palate/palate.toml`, defaults. Unknown keys are an
error at startup rather than a silent no-op.

## What works

`palate ingest` reads the five CSVs out of the export, reconciles them per
Letterboxd URI, and writes `user_films`. Ratings are stored as half-star
integers 1 to 10 so nothing downstream compares floats. Rows that cannot be
matched to a TMDB id are kept in `unmatched_export_row` with a reason, because
dropping them quietly would bias every later measurement toward mainstream
titles.

`palate tmdb crawl` fills `films`, `people` and `credits` from TMDB. The queue is
the only state, so killing it and rerunning finishes the set rather than starting
again. Requests are paced at twenty per second and back off on a 429. Every raw
payload is kept zlib compressed, about 200 MB for forty thousand films, so
`palate tmdb renormalize` can rebuild every derived row in a minute without
touching the API.

## What does not work yet

Corpus eligibility is not applied yet, so every crawled film counts as a member.
The discover sweep pages a single popularity query, which cannot reach forty
thousand films on its own. The embedding index, the taste model, the agent and
the evaluation harness are not built, and the CLI so far is `palate ingest`,
`palate tmdb` and `palate version`.

## License

MIT.
