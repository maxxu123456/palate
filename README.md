# palate

A film recommender built on one person's Letterboxd history. It all runs on the laptop:
corpus and embeddings in one SQLite file, chat model in this process. Fitted to one viewer.

## Setup

```sh
uv sync && cp .env.example .env
mkdir -p ~/.config/palate && cp palate.toml.example ~/.config/palate/palate.toml
```

Nothing to pick, and no daemon or key. Chat is a transformers pipeline, embeddings are
sentence-transformers, reranking is a cross-encoder, all three on MPS and pulled on first use:

- chat, Qwen/Qwen2.5-3B-Instruct, 6.2 GB
- embeddings, google/embeddinggemma-300m, 1.3 GB, gated, so it wants `HF_TOKEN`
- rerank, cross-encoder/ms-marco-MiniLM-L6-v2, 90 MB

`models.toml` pins each alias to a Hub commit sha, since a moved revision quietly invalidates
every index built from it. Nothing resolves a sha at runtime, so edit that file to move a pin.

## Usage

```sh
palate ingest ~/Downloads/letterboxd-2026-09-14.zip  # letterboxd settings, data, export
palate tmdb crawl  # reads TMDB_READ_TOKEN from .env
palate index build
palate profile build
palate recommend "something slow and cold but not russian" -n 5
palate chat "what should I watch tonight, nothing russian"
palate serve  # the same loop over http on 127.0.0.1:8000, needs --extra api
```

`chat` is a hand-rolled loop over ten tools. The model returns ids, never titles: the prose
is assembled from database rows and every sentence is checked against what the run retrieved.
Seven guards end a run and all of them still answer. `palate traces show <run>` prints the
span tree, from a file of hashes unless `trace.payloads` is full.

## Evaluation

The holdout is temporal: a backlog import stamps thousands of films on one afternoon,
so any day over 2 percent of the history is train-only in every fold. Results go to
`eval/report.md`, arm by arm against popularity and director baselines.

```sh
palate eval split build && palate eval run && make report
```

## Status

Works: ingest, crawl, corpus, index, taste profile, retrieval, eval harness, rerank, agent,
cli chat, http api, listing ui. No collaborative filtering, and exact search stops near
100k vectors.

TODO: the chat panel is behind VITE_PALATE_CHAT and does not collapse tool traces yet.
TODO: the rating lora under experiments/ is one unreproduced run. It needs a fold aware
split before it means anything.

## License

MIT.
