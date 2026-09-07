# palate

A film recommender built on one person's Letterboxd history. Everything runs on
the laptop: the corpus is a SQLite file, the embeddings live in the same file,
and the chat model is whatever you point it at. The interesting part is not the
chat, it is that the ranking is fitted to one viewer's ratings instead of to a
crowd.

## Setup

```sh
uv sync
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

Then export your Letterboxd data (Settings, Data, Export your data) and import it:

```sh
palate doctor
palate ingest ~/Downloads/letterboxd-2026-09-14.zip
palate ingest review
```

## Configuration

Precedence, highest first: CLI flag, `PALATE_` environment variables, `.env`,
`./palate.toml`, `~/.config/palate/palate.toml`, defaults. Unknown keys are an
error at startup rather than a silent no-op.

```sh
palate config show
```

## What works

`palate ingest` reads the five CSVs out of the export, reconciles them per
Letterboxd URI, and writes `user_films`. Ratings are stored as half-star
integers 1 to 10 so nothing downstream compares floats. Rows that cannot be
matched to a TMDB id are kept in `unmatched_export_row` with a reason, because
dropping them quietly would bias every later measurement toward mainstream
titles.

## What does not work yet

Title resolution needs a TMDB search backend and there is not one wired in, so a
fresh import resolves only the URIs that already carry a TMDB id and queues the
rest for review. The corpus crawl, the embedding index, the taste model, the
agent and the evaluation harness are not built.

## License

MIT.
