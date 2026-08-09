import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import html as html_module
    from io import BytesIO

    import marimo as mo
    import polars as pl

    from detexify_data import load_detexify, load_manifest, rasterize_strokes

    return (
        BytesIO,
        base64,
        html_module,
        load_detexify,
        load_manifest,
        mo,
        pl,
        rasterize_strokes,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Detexify handwriting dataset

    Explore the [official Detexify data export](https://github.com/kirel/detexify-data):
    native pen strokes for handwritten LaTeX symbols, joined to their command and
    package metadata. The upstream database is licensed under the
    [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

    On its first run, this notebook downloads the pinned official archive (about
    204 MiB), validates it, and creates a reusable partitioned Parquet cache under
    `data/detexify/`. Later runs read that cache directly.
    """)
    return


@app.cell
def _(load_detexify, load_manifest):
    detexify_df = load_detexify()
    detexify_manifest = load_manifest()
    return detexify_df, detexify_manifest


@app.cell
def _(detexify_df, detexify_manifest, mo):
    mo.md(
        f"""
        ## Import complete

        Loaded **{detexify_df.height:,} labeled samples** across
        **{detexify_manifest["class_count"]:,} symbols**. The importer validated
        all {detexify_manifest["raw_rows"]:,} source rows; rows with no symbol
        label omitted: {detexify_manifest["skipped_unlabeled_rows"]}.
        """
    )
    return


@app.cell
def _(detexify_df, mo, pl):
    class_sizes = detexify_df.group_by("symbol_id").len().get_column("len")
    dataset_summary = pl.DataFrame(
        {
            "metric": [
                "samples",
                "classes",
                "minimum samples per class",
                "median samples per class",
                "maximum samples per class",
                "total strokes",
                "total points",
            ],
            "value": [
                detexify_df.height,
                class_sizes.len(),
                class_sizes.min(),
                int(class_sizes.median()),
                class_sizes.max(),
                detexify_df.get_column("stroke_count").sum(),
                detexify_df.get_column("point_count").sum(),
            ],
        }
    )
    schema_summary = pl.DataFrame(
        {
            "column": detexify_df.schema.names(),
            "dtype": [str(dtype) for dtype in detexify_df.schema.dtypes()],
        }
    )
    mo.vstack(
        [
            mo.md("## Dataset statistics"),
            dataset_summary,
            mo.md("## Polars schema"),
            schema_summary,
        ],
        gap=1,
    )
    return


@app.cell
def _(BytesIO, base64, detexify_df, html_module, mo, rasterize_strokes):
    gallery_samples = detexify_df.sample(n=12, seed=0).select(
        "command",
        "package",
        "stroke_count",
        "point_count",
        "strokes",
    )
    gallery_cards = []
    for gallery_sample in gallery_samples.iter_rows(named=True):
        raster_image = rasterize_strokes(gallery_sample["strokes"])
        image_buffer = BytesIO()
        raster_image.save(image_buffer, format="PNG")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        command_label = html_module.escape(gallery_sample["command"])
        package_label = html_module.escape(gallery_sample["package"])
        gallery_cards.append(
            f"""
            <figure class="detexify-card">
              <img src="data:image/png;base64,{image_data}"
                   alt="Handwritten {command_label}">
              <figcaption>
                <code>{command_label}</code><br>
                <span>{package_label}</span><br>
                <small>{gallery_sample["stroke_count"]} strokes ·
                {gallery_sample["point_count"]} points</small>
              </figcaption>
            </figure>
            """
        )
    gallery_html = mo.Html(
        """
        <style>
          .detexify-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
          }
          .detexify-card {
            margin: 0;
            padding: 0.8rem;
            border: 1px solid var(--sl-color-gray-5);
            border-radius: 0.6rem;
            text-align: center;
          }
          .detexify-card img {
            width: 96px;
            height: 96px;
            image-rendering: auto;
            border: 1px solid var(--sl-color-gray-6);
          }
          .detexify-card figcaption { margin-top: 0.5rem; }
          .detexify-card span, .detexify-card small {
            color: var(--sl-color-gray-3);
          }
        </style>
        <div class="detexify-grid">
        """
        + "".join(gallery_cards)
        + "</div>"
    )
    mo.vstack([mo.md("## Deterministic sample gallery"), gallery_html], gap=1)
    return


@app.cell
def _(detexify_df):
    class_counts = (
          detexify_df
          .group_by(["command", "package"])
          .len()
          .rename({"len": "sample_count"})
          .sort("sample_count", descending=True)
      )
    class_counts
    return


if __name__ == "__main__":
    app.run()
