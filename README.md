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

## Data license

The Detexify database is published by Daniel Kirsch under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
This repository's code is licensed separately; see [LICENSE](LICENSE).
