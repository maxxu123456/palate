# palate

A film recommender built on one person's Letterboxd history. Everything runs on
the laptop: the corpus is a SQLite file, the embeddings live in the same file,
and the chat model is whatever you point it at. The ranking is fitted to one
viewer's ratings rather than to a crowd.

## Setup

```sh
uv sync
mkdir -p ~/.config/palate
cp palate.toml.example ~/.config/palate/palate.toml
cp .env.example .env
brew install ollama && ollama serve && ollama pull qwen3:8b  # default provider
```

For a hosted model set `provider = "openrouter"` and put the key name in
`api_key_env`. Any OpenAI-compatible `base_url` works too (LM Studio, vLLM,
Together, Groq). Embeddings are a separate provider, because most chat hosts do
not serve them: `uv sync --extra local` runs google/embeddinggemma-300m on MPS
with no daemon. Secrets stay out of the config file, which holds only the name of
the variable they live in. A TMDB v4 read token goes in `TMDB_READ_TOKEN`, and
without it titles resolve only from Letterboxd URIs that carry a TMDB id.

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

The holdout is temporal. A Letterboxd export dates a rating when it was entered,
so a backlog import stamps thousands of films on one afternoon, and any calendar
day holding more than 2 percent of the history is train-only in every fold.

```sh
palate eval split build
palate eval run
make readme-table
```

What sits below is the planted history the tests run on, 240 ratings over two
folds. Every interval spans zero and `ns` says so, so read it as the shape of the
table and not as a result. Run the three commands on your own export.

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

No reranker, so every arm ranks with stage one only and the cross-encoder rows
print their reason instead of a number. No agent, no HTTP surface. No item-item
collaborative filtering, and there never will be: one user, no co-rating matrix,
nothing to collaborate with. Nearest neighbour search is exact, which stops being
fine somewhere past a few hundred thousand vectors.

## License

MIT.
