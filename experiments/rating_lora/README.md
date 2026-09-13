# rating lora

Does a small learned head order the pool better than the ridge direction the taste
model already fits? The ridge sits on hand built features. This asks a base model to
read the same metadata as text and predict the star rating instead.

Nothing in here is imported by palate. It is an experiment, not a feature, and the
package runs with this directory deleted.

## Dataset

```sh
python dataset.py --db ~/.palate/palate.db --out data/ratings.jsonl
```

One line per rated film: the tmdb id, the metadata block the model reads, the rating
in stars, and the split. The rating is the label so it never appears in the text, and
review text stays out because it is the viewer writing about the rating.

## Where it stands

0.71 MAE on the validation split against a 0.78 mean baseline. That is one run on a
rented GPU and it is not a result yet. The split is random, so films logged on the
same catalogue day land on both sides of it, and the eval harness already knows how
to fold on reliable dates instead. Until this splits the same way the number is
measuring a leak as much as a model.
