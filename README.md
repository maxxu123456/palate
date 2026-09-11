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

Ollama is the default chat provider and needs no key:

```sh
brew install ollama && ollama serve && ollama pull qwen3:8b
```

For a hosted model set `provider = "openrouter"` and put the key name in
`api_key_env`. Any OpenAI-compatible `base_url` works too (LM Studio, vLLM,
Together, Groq). Embeddings are a separate provider, because most chat hosts do
not serve them: `uv sync --extra local` runs google/embeddinggemma-300m on MPS
with no daemon at all. Secrets never go in the config file, only the name of the
environment variable that holds them.

A TMDB v4 read token goes in `TMDB_READ_TOKEN`. Without it titles resolve only
from Letterboxd URIs that already carry a TMDB id.

## Usage

Export your Letterboxd data (Settings, Data, Export your data), then:

```sh
palate ingest ~/Downloads/letterboxd-2026-09-14.zip
palate tmdb crawl
palate index build
palate profile build
palate recommend "something slow and cold but not russian" -n 5
```

## Evaluation

The holdout is temporal and refuses to pretend otherwise. A Letterboxd export
dates a rating when it was entered, so a backlog import stamps thousands of films
on one afternoon and a naive split scores that shuffle as if it were the future.
Any calendar day holding more than 2 percent of the history is train-only in
every fold, and under 300 reliably dated films the harness refuses the temporal
protocol and prints the refusal above the table.

```sh
palate eval split build
palate eval run
make readme-table
```

<!-- eval-table:start -->

No numbers here yet. Run the three commands above against your own history and
`make readme-table` writes the generated table into this spot, including the rows
where a stage did not help.

<!-- eval-table:end -->

## What does not work yet

No reranker, so every arm ranks with stage one only and the cross-encoder rows
print their reason instead of a number. No agent, no HTTP surface. No item-item
collaborative filtering, and there never will be: one user, no co-rating matrix,
nothing to collaborate with. Nearest neighbour search is exact, which is fine to
a few hundred thousand vectors and stops being fine after that.

## License

MIT.
