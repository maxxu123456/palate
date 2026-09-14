# palate

A film recommender built on one person's Letterboxd history. Everything runs on
the laptop: the corpus is a SQLite file, the embeddings live in the same file,
and the chat model loads into the same process. Fitted to one viewer, not a crowd.

## Setup

```sh
uv sync && cp .env.example .env
mkdir -p ~/.config/palate && cp palate.toml.example ~/.config/palate/palate.toml
```

Everything is Hugging Face and everything is local: no daemon, no inference
key. Chat is a transformers pipeline, embeddings are sentence-transformers,
reranking is a cross-encoder, all three on MPS in this process. The first run
of each pulls its pinned snapshot, about 7 GB together. Embeddings stay a provider
of their own. TMDB needs `TMDB_READ_TOKEN`, a gated repo needs `HF_TOKEN`.

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

`chat` is a hand-rolled loop over ten tools. The model returns ids, never titles: the
prose is assembled from database rows and every sentence is checked against what the
run retrieved. Seven guards end a run and all of them still answer. `palate traces show
<run>` prints the span tree, from a file of hashes unless `trace.payloads` is full.

## Evaluation

The holdout is temporal: a backlog import stamps thousands of films on one afternoon,
so any day over 2 percent of the history is train-only in every fold. 240 planted
ratings over two folds, and every interval spans zero, so read the shape not the rank.

```sh
palate eval split build && palate eval run && make readme-table
```

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

## Status

Works: ingest, crawl, corpus, index, taste profile, retrieval, eval harness, rerank,
agent, cli chat, http api, listing ui. The table above is the committed fold results.

Both rerankers are eval arms and an arm only runs where its checkpoint already is, so
`eval report` prints their reason instead of a number until the weights are pulled. The
checker's paraphrase arm is not wired up, so `grounded_ratio` has no measured precision
behind it. No collaborative filtering. Exact search stops near 100k vectors.

TODO: the chat panel is behind VITE_PALATE_CHAT and does not collapse tool traces yet.
TODO: the rating lora under experiments/ is one unreproduced run, 0.71 MAE against a
0.78 baseline. Needs a fold aware split before it means anything.

## License

MIT.
