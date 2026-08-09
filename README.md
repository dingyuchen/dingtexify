# dingtexify

A [Marimo](https://marimo.io/) notebook for importing and exploring the
[official Detexify handwriting dataset](https://github.com/kirel/detexify-data).
The loader preserves each native pen stroke in a typed Polars DataFrame and
provides deterministic 64×64 grayscale rasterization for image-model workflows.

## Run it

Install the locked dependencies and open the notebook:

```sh
uv sync
uv run marimo edit notebook.py
```

Or run the notebook as a read-only app:

```sh
uv run marimo run notebook.py
```

The first call to `load_detexify()` automatically downloads the pinned official
SQL export and symbol metadata (about 204 MiB compressed). It verifies both
SHA-256 hashes, streams the gzip-compressed PostgreSQL `COPY` section into Zstd
Parquet parts, and stores them under `data/detexify/`. The decompressed SQL dump
is never written to disk. Later calls validate and reuse the manifest-backed
cache.

The completed import contains 210,454 labeled samples across 1,098 symbols; all
210,454 rows in the pinned `samples` export are labeled. The importer still
tracks and reports unlabeled rows so a future export cannot silently change the
dataset. Malformed strokes, unknown symbols, changed hashes, and changed official
counts fail with an explicit error.

## Python API

```python
from detexify_data import load_detexify, rasterize_strokes

samples = load_detexify()
image = rasterize_strokes(samples.row(0, named=True)["strokes"], size=64)
```

`load_detexify()` returns an eager `polars.DataFrame` with sample and symbol
metadata, native `List[List[Struct{x, y, t}]]` strokes, and precomputed stroke
and point counts. Raster images are generated lazily and are not duplicated in
the DataFrame.

## Vision Transformer classifier

The classifier in [`classifier.py`](classifier.py) accepts a batch shaped
`(batch, 1, 32, 32)`. It divides each grayscale image into 64 non-overlapping
4×4 patches, projects them into tokens, and passes them through four Transformer
encoder blocks. A learned classification token produces logits over the
dataset's distinct LaTeX commands. Records with the same command but different
package identifiers intentionally share one output class.

Train it on the full dataset with a stratified validation split and class-
weighted loss:

```sh
uv run python train_classifier.py \
  --epochs 30 \
  --batch-size 128 \
  --output checkpoints/detexify-vit.pt
```

CUDA is preferred automatically, followed by Apple Metal (MPS) and CPU. Images
are rasterized lazily from the native strokes, and the best validation checkpoint
contains both the weights and exact command-to-index vocabulary.

Load a checkpoint and classify a black-on-white 32×32 raster:

```python
from PIL import Image

from classifier import load_classifier_checkpoint, predict_commands

model, vocabulary, metadata = load_classifier_checkpoint(
    "checkpoints/detexify-vit.pt"
)
image = Image.open("symbol.png")

for prediction in predict_commands(model, vocabulary, image, top_k=5):
    print(prediction.command, prediction.probability)
```

The returned probabilities are ranked from most to least likely. Training saves
top-1 and top-5 validation accuracy in `metadata` for later comparison.

## Data license

The Detexify database is published by Daniel Kirsch under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
This repository's code is licensed separately; see [LICENSE](LICENSE).
