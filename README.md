# palate

A film recommender built on one person's Letterboxd history. Everything runs on
the laptop: the corpus is a SQLite file, the embeddings live in the same file,
and the chat model is whatever you point it at. Fitted to one viewer, not a crowd.

## Setup

```sh
uv sync && cp .env.example .env
mkdir -p ~/.config/palate && cp palate.toml.example ~/.config/palate/palate.toml
brew install ollama && ollama serve && ollama pull qwen3:8b  # default provider
```

For a hosted model set `provider = "openrouter"` and the key name in
`api_key_env`. Any OpenAI-compatible `base_url` works too. Embeddings are a
separate provider because most chat hosts do not serve them: `uv sync --extra
local` runs google/embeddinggemma-300m on MPS. Keys live in env vars, never in
the config. A TMDB v4 token goes in `TMDB_READ_TOKEN`.

## Usage

Export your Letterboxd data (Settings, Data, Export your data), then:

```sh
palate ingest ~/Downloads/letterboxd-2026-09-14.zip
palate tmdb crawl
palate index build
palate profile build
palate recommend "something slow and cold but not russian" -n 5
palate chat "what should I watch tonight, nothing russian"
palate serve  # the same loop over http on 127.0.0.1:8000, needs --extra api
```

`chat` is a hand-rolled loop over ten tools. The model returns ids, never titles:
the prose is assembled from database rows and every sentence is checked against
what the run retrieved. Seven guards end a run and all of them still answer.
`palate traces show <run>` prints the span tree with tokens and cost, from a
second file holding hashes rather than text unless `trace.payloads = "full"`.

## Evaluation

The holdout is temporal. A Letterboxd export dates a rating when it was entered,
so a backlog import stamps thousands of films on one afternoon, and any day over
2 percent of the history is train-only in every fold.

```sh
palate eval split build && palate eval run && make readme-table
```

240 planted ratings over two folds. Every interval spans zero: read the shape.

<!-- eval-table:start -->

| arm | ndcg@10 | 95% CI | recall@50 | d vs director_affinity |
|---|---|---|---|---|
| popularity | 0.033 | 0.00-0.03 | 0.139 | -0.143 ns |
| director_affinity | 0.177 | 0.00-0.18 | 0.407 |  |
| single_centroid_dense | 0.258 | 0.04-0.27 | 0.461 | +0.082 ns |
| dense_only | 0.227 | 0.03-0.23 | 0.504 | +0.050 ns |
| +ridge +repulsion | 0.311 | 0.04-0.32 | 0.479 | +0.134 ns |
| +exposure_features | 0.404 | 0.04-0.41 | 0.443 | +0.228 ns |
| +people_priors | 0.248 | 0.00-0.26 | 0.461 | +0.071 ns |
| full | 0.123 | 0.00-0.12 | 0.479 | -0.054 ns |

<!-- eval-table:end -->

## What does not work yet

Both rerankers are eval arms and an arm only runs where its checkpoint is, so
`eval report` prints their reason instead of a number. `uv sync --extra local`
then `palate eval run` puts them against each other on NDCG, milliseconds and
dollars at once. That extra is also the checker's paraphrase arm, without which
a reworded plot line is marked unsupported, and the checker has no measured
precision yet, so `grounded_ratio` is not a number to quote. No collaborative
filtering. Neighbour search is exact and stops scaling past 100k vectors.

## License

MIT.
