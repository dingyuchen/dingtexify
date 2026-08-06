from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import httpx
import polars as pl
import pytest

import detexify_data as detexify

SYMBOL_ID = "latex2e-OT1-_alpha"
SECOND_SYMBOL_ID = "amsfonts-OT1-_mathbb{A}"


def _write_fixture_sources(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    sql_path = directory / "fixture.sql.gz"
    rows = [
        (
            "1",
            SYMBOL_ID,
            [[[0, 0, 100], [10, 20, 110]], [[10, 20, 111], [20, 0, 120]]],
        ),
        ("2", "", [[[5, 5, 200]]]),
        ("3", SECOND_SYMBOL_ID, [[[2.5, 7, 300], [12.5, 7, 310]]]),
    ]
    sql_lines = [
        "-- fixture header",
        "COPY samples (id, key, strokes) FROM stdin;",
        *[
            f"{sample_id}\t{symbol_id}\t{json.dumps(strokes, separators=(',', ':'))}"
            for sample_id, symbol_id, strokes in rows
        ],
        r"\.",
        "-- fixture footer",
    ]
    with gzip.open(sql_path, "wt", encoding="utf-8", newline="") as output:
        output.write("\n".join(sql_lines) + "\n")

    symbols_path = directory / "symbols.json"
    symbols_path.write_text(
        json.dumps(
            [
                {
                    "id": SYMBOL_ID,
                    "command": r"\alpha",
                    "mathmode": True,
                    "textmode": False,
                },
                {
                    "id": SECOND_SYMBOL_ID,
                    "command": r"\mathbb{A}",
                    "package": "amsfonts",
                    "fontenc": "T1",
                    "mathmode": True,
                    "textmode": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    return sql_path, symbols_path


def test_parse_google_drive_confirmation_form() -> None:
    html = """
    <html><form action="https://drive.usercontent.google.com/download">
      <input type="hidden" name="id" value="file-id">
      <input type="hidden" name="confirm" value="t">
      <input type="hidden" name="uuid" value="one-time-token">
    </form></html>
    """

    url, parameters = detexify._parse_download_form(html, "https://drive.google.com/uc")

    assert url == "https://drive.usercontent.google.com/download"
    assert parameters == {
        "id": "file-id",
        "confirm": "t",
        "uuid": "one-time-token",
    }


def test_download_file_follows_warning_form_and_is_atomic(tmp_path: Path) -> None:
    payload = b"detexify fixture payload"
    spec = detexify.DownloadSpec(
        name="fixture.bin",
        file_id="fixture-id",
        resource_key="fixture-key",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "drive.google.com":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="""
                <form action="https://drive.usercontent.google.com/download">
                  <input name="id" value="fixture-id">
                  <input name="confirm" value="t">
                  <input name="uuid" value="token">
                </form>
                """,
                request=request,
            )
        assert request.url.params["confirm"] == "t"
        return httpx.Response(200, content=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    destination = tmp_path / spec.name

    assert detexify.download_file(spec, destination, client=client) == destination
    assert destination.read_bytes() == payload
    assert not (tmp_path / "fixture.bin.part").exists()
    assert len(requests) == 2


def test_download_hash_failure_leaves_no_artifact(tmp_path: Path) -> None:
    payload = b"corrupt"
    spec = detexify.DownloadSpec(
        name="fixture.bin",
        file_id="fixture-id",
        resource_key="fixture-key",
        sha256=hashlib.sha256(b"expected").hexdigest(),
        size=len(payload),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=payload,
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    destination = tmp_path / spec.name

    with pytest.raises(detexify.DatasetIntegrityError, match="SHA-256"):
        detexify.download_file(spec, destination, client=client)

    assert not destination.exists()
    assert not (tmp_path / "fixture.bin.part").exists()


def test_sql_parser_preserves_native_strokes(tmp_path: Path) -> None:
    sql_path, _ = _write_fixture_sources(tmp_path)

    rows = list(detexify._iter_sql_samples(sql_path))

    assert [row[:2] for row in rows] == [
        (1, SYMBOL_ID),
        (2, ""),
        (3, SECOND_SYMBOL_ID),
    ]
    assert rows[0][2][0][1] == {"x": 10.0, "y": 20.0, "t": 110}
    assert rows[2][2][0][0]["x"] == 2.5


def test_build_cache_joins_metadata_and_reports_unlabeled_rows(
    tmp_path: Path,
) -> None:
    sql_path, symbols_path = _write_fixture_sources(tmp_path / "source")
    processed = tmp_path / "processed"

    manifest = detexify._build_cache(
        sql_path,
        symbols_path,
        processed,
        batch_size=1,
        validate_official_counts=False,
    )
    frame = pl.read_parquet([processed / part for part in manifest["parts"]])

    assert manifest["raw_rows"] == 3
    assert manifest["output_rows"] == 2
    assert manifest["skipped_unlabeled_rows"] == 1
    assert manifest["class_count"] == 2
    assert frame.schema == detexify.DETEXIFY_SCHEMA
    assert frame.get_column("sample_id").to_list() == [1, 3]
    assert frame.row(0, named=True)["package"] == "latex2e"
    assert frame.row(0, named=True)["font_encoding"] == "OT1"
    assert frame.row(1, named=True)["package"] == "amsfonts"
    assert frame.row(1, named=True)["font_encoding"] == "T1"
    assert frame.get_column("stroke_count").to_list() == [2, 1]
    assert frame.get_column("point_count").to_list() == [4, 2]


def test_load_reuses_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql_path, symbols_path = _write_fixture_sources(tmp_path / "source")
    data_dir = tmp_path / "data"
    processed = data_dir / detexify.CACHE_DIRECTORY
    detexify._build_cache(
        sql_path,
        symbols_path,
        processed,
        validate_official_counts=False,
    )
    monkeypatch.setattr(detexify, "EXPECTED_OUTPUT_ROWS", 2)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid cache should avoid downloading")

    monkeypatch.setattr(detexify, "_ensure_sources", fail_if_called)

    assert detexify.load_detexify(data_dir).height == 2


def _dark_bounds(image: object) -> tuple[int, int, int, int]:
    grayscale = image  # Pillow image; kept untyped to avoid importing private aliases.
    dark = [
        (x, y)
        for y in range(grayscale.height)
        for x in range(grayscale.width)
        if grayscale.getpixel((x, y)) < 128
    ]
    return (
        min(x for x, _ in dark),
        min(y for _, y in dark),
        max(x for x, _ in dark),
        max(y for _, y in dark),
    )


def test_rasterizer_is_grayscale_sized_and_deterministic() -> None:
    strokes = [[{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 20.0}]]

    first = detexify.rasterize_strokes(strokes)
    second = detexify.rasterize_strokes(strokes)

    assert first.mode == "L"
    assert first.size == (64, 64)
    assert first.tobytes() == second.tobytes()
    left, top, right, bottom = _dark_bounds(first)
    assert 16 <= left <= 21
    assert 4 <= top <= 9
    assert 42 <= right <= 47
    assert 54 <= bottom <= 59
    assert 0.45 <= (right - left) / (bottom - top) <= 0.60


def test_rasterizer_handles_degenerate_axes_and_single_points() -> None:
    vertical = detexify.rasterize_strokes([[[4, 0, 1], [4, 20, 2]]])
    point = detexify.rasterize_strokes([[[9, 9, 1]]])

    vertical_left, _, vertical_right, _ = _dark_bounds(vertical)
    point_left, point_top, point_right, point_bottom = _dark_bounds(point)
    assert 29 <= (vertical_left + vertical_right) / 2 <= 34
    assert 29 <= (point_left + point_right) / 2 <= 34
    assert 29 <= (point_top + point_bottom) / 2 <= 34


@pytest.mark.skipif(
    os.environ.get("DETEXIFY_FULL_TEST") != "1",
    reason="set DETEXIFY_FULL_TEST=1 to download and validate the official export",
)
def test_official_dataset_end_to_end(tmp_path: Path) -> None:
    data_dir = Path(os.environ.get("DETEXIFY_DATA_DIR", tmp_path / "detexify"))
    frame = detexify.load_detexify(data_dir)
    manifest = detexify.load_manifest(data_dir)

    assert frame.height == detexify.EXPECTED_OUTPUT_ROWS
    assert frame.get_column("symbol_id").n_unique() == detexify.EXPECTED_CLASSES
    assert manifest["raw_rows"] == detexify.EXPECTED_RAW_ROWS
    assert manifest["skipped_unlabeled_rows"] == detexify.EXPECTED_UNLABELED_ROWS
