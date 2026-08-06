"""Download and load the official Detexify handwriting dataset.

The upstream dataset is a PostgreSQL text dump containing native pen strokes.
This module streams its ``samples`` COPY section into partitioned Parquet so
that the one-gigabyte decompressed SQL file is never materialized on disk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import polars as pl
from PIL import Image, ImageDraw

DATASET_VERSION = 1
CACHE_DIRECTORY = f"processed-v{DATASET_VERSION}"
EXPECTED_RAW_ROWS = 210_454
EXPECTED_OUTPUT_ROWS = 210_454
EXPECTED_UNLABELED_ROWS = 0
EXPECTED_CLASSES = 1_098
EXPECTED_MIN_CLASS_SIZE = 4
EXPECTED_MAX_CLASS_SIZE = 3_937


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    file_id: str
    resource_key: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return (
            "https://drive.google.com/uc?export=download"
            f"&id={self.file_id}&resourcekey={self.resource_key}"
        )


SQL_DUMP = DownloadSpec(
    name="detexify.sql.gz",
    file_id="0ByuYordD0JBRV01NM2pmNlpfNUE",
    resource_key="0-CZHt-PBM7v0hty25FF5wsg",
    sha256="a270abcd031582364f6ede1a5dac3f3c3a6cd2d15c2b61f7bb2726b0f716d128",
    size=213_990_614,
)
SYMBOLS = DownloadSpec(
    name="symbols.json",
    file_id="0ByuYordD0JBRU1Y3Q3VSNk9kdE0",
    resource_key="0-V2m8tmPfD8eyNe4GGrhSxw",
    sha256="13bfd78164d92f4fb73856926fcd2c272ec171b5c35110de76c4ec6c0cdfced0",
    size=171_013,
)
DOWNLOADS = (SQL_DUMP, SYMBOLS)


type Point = dict[str, float | int]
type Stroke = list[Point]
type Strokes = list[Stroke]
type ProgressCallback = Callable[[str, int, int | None], None]

POINT_DTYPE = pl.Struct(
    {
        "x": pl.Float64,
        "y": pl.Float64,
        "t": pl.Int64,
    }
)
DETEXIFY_SCHEMA = pl.Schema(
    {
        "sample_id": pl.Int64,
        "symbol_id": pl.String,
        "command": pl.String,
        "package": pl.String,
        "font_encoding": pl.String,
        "math_mode": pl.Boolean,
        "text_mode": pl.Boolean,
        "strokes": pl.List(pl.List(POINT_DTYPE)),
        "stroke_count": pl.UInt16,
        "point_count": pl.UInt32,
    }
)
SCHEMA_DESCRIPTION = {name: str(dtype) for name, dtype in DETEXIFY_SCHEMA.items()}


class DetexifyError(RuntimeError):
    """Base class for Detexify import errors."""


class DatasetIntegrityError(DetexifyError):
    """Raised when a downloaded or cached artifact fails validation."""


class DatasetFormatError(DetexifyError):
    """Raised when an upstream artifact does not match its documented format."""


class _DownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.forms: list[tuple[str, dict[str, str]]] = []
        self._action: str | None = None
        self._inputs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form":
            self._action = attributes.get("action")
            self._inputs = {}
        elif tag == "input" and self._action is not None:
            name = attributes.get("name")
            value = attributes.get("value")
            if name is not None and value is not None:
                self._inputs[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._action is not None:
            self.forms.append((self._action, self._inputs.copy()))
            self._action = None
            self._inputs = {}


def _parse_download_form(html: str, base_url: str) -> tuple[str, dict[str, str]]:
    parser = _DownloadFormParser()
    parser.feed(html)
    for action, parameters in parser.forms:
        if "download" in action or "confirm" in parameters:
            return urljoin(base_url, action), parameters
    raise DatasetFormatError(
        "Google Drive returned HTML without the expected download confirmation form"
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_response(
    response: httpx.Response,
    destination: Path,
    spec: DownloadSpec,
    progress: ProgressCallback | None,
) -> str:
    response.raise_for_status()
    digest = hashlib.sha256()
    downloaded = 0
    content_length = response.headers.get("content-length")
    total = (
        int(content_length)
        if content_length and content_length.isdigit()
        else spec.size
    )
    with destination.open("wb") as output:
        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if progress is not None:
                progress(f"download:{spec.name}", downloaded, total)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def download_file(
    spec: DownloadSpec,
    destination: Path,
    *,
    client: httpx.Client | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download one Drive artifact atomically and verify its pinned SHA-256."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual_hash = _sha256(destination)
        if actual_hash == spec.sha256:
            return destination
        raise DatasetIntegrityError(
            f"Existing {destination} has SHA-256 {actual_hash}, expected {spec.sha256}. "
            "Remove that file to allow a clean download."
        )

    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            follow_redirects=True,
            timeout=None,
            headers={"User-Agent": "dingtexify/0.1 dataset importer"},
        )

    try:
        with client.stream("GET", spec.url) as first_response:
            first_response.raise_for_status()
            content_type = first_response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                form_url, parameters = _parse_download_form(
                    first_response.read().decode("utf-8", errors="replace"),
                    str(first_response.url),
                )
                with client.stream("GET", form_url, params=parameters) as response:
                    actual_hash = _write_response(response, partial, spec, progress)
            else:
                actual_hash = _write_response(first_response, partial, spec, progress)

        if actual_hash != spec.sha256:
            raise DatasetIntegrityError(
                f"Downloaded {spec.name} has SHA-256 {actual_hash}, "
                f"expected {spec.sha256}"
            )
        if partial.stat().st_size != spec.size:
            raise DatasetIntegrityError(
                f"Downloaded {spec.name} is {partial.stat().st_size:,} bytes, "
                f"expected {spec.size:,}"
            )
        partial.replace(destination)
        return destination
    finally:
        if owns_client:
            client.close()
        if partial.exists():
            partial.unlink()


def _ensure_sources(
    data_dir: Path, progress: ProgressCallback | None
) -> dict[str, Path]:
    raw_dir = data_dir / "raw"
    return {
        spec.name: download_file(
            spec,
            raw_dir / spec.name,
            progress=progress,
        )
        for spec in DOWNLOADS
    }


def _load_symbols(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"Could not parse {path}: {error}") from error
    if not isinstance(payload, list):
        raise DatasetFormatError("symbols.json must contain a JSON array")

    symbols: dict[str, dict[str, Any]] = {}
    for index, symbol in enumerate(payload):
        if not isinstance(symbol, dict) or not isinstance(symbol.get("id"), str):
            raise DatasetFormatError(f"Invalid symbol metadata at index {index}")
        symbol_id = symbol["id"]
        if symbol_id in symbols:
            raise DatasetFormatError(f"Duplicate symbol id {symbol_id!r}")
        symbols[symbol_id] = symbol
    return symbols


def _normalize_strokes(value: Any, sample_id: int) -> Strokes:
    if not isinstance(value, list) or not value:
        raise DatasetFormatError(f"Sample {sample_id} has no stroke list")
    normalized: Strokes = []
    for stroke_index, stroke in enumerate(value):
        if not isinstance(stroke, list) or not stroke:
            raise DatasetFormatError(
                f"Sample {sample_id}, stroke {stroke_index} is empty or invalid"
            )
        normalized_stroke: Stroke = []
        for point_index, point in enumerate(stroke):
            if not isinstance(point, list) or len(point) != 3:
                raise DatasetFormatError(
                    f"Sample {sample_id}, stroke {stroke_index}, point "
                    f"{point_index} is not [x, y, t]"
                )
            x, y, timestamp = point
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or isinstance(timestamp, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(x))
                or not math.isfinite(float(y))
                or not math.isfinite(float(timestamp))
                or float(timestamp) != int(timestamp)
            ):
                raise DatasetFormatError(
                    f"Sample {sample_id} contains a non-numeric point"
                )
            normalized_stroke.append(
                {"x": float(x), "y": float(y), "t": int(timestamp)}
            )
        normalized.append(normalized_stroke)
    return normalized


def _iter_sql_samples(path: Path) -> Iterator[tuple[int, str, Strokes]]:
    copy_header = "COPY samples (id, key, strokes) FROM stdin;"
    in_samples = False
    found_header = False
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.rstrip("\r\n")
                if not in_samples:
                    if line == copy_header:
                        in_samples = True
                        found_header = True
                    continue
                if line == r"\.":
                    return
                fields = line.split("\t", 2)
                if len(fields) != 3:
                    raise DatasetFormatError(
                        f"Malformed samples row at SQL line {line_number}"
                    )
                sample_text, symbol_id, strokes_text = fields
                try:
                    sample_id = int(sample_text)
                    strokes_value = json.loads(strokes_text)
                except (ValueError, json.JSONDecodeError) as error:
                    raise DatasetFormatError(
                        f"Malformed samples row at SQL line {line_number}: {error}"
                    ) from error
                yield sample_id, symbol_id, _normalize_strokes(strokes_value, sample_id)
    except (OSError, EOFError) as error:
        raise DatasetFormatError(f"Could not read {path}: {error}") from error
    if not found_header:
        raise DatasetFormatError(f"Did not find {copy_header!r} in {path}")
    raise DatasetFormatError("The samples COPY section has no terminator")


def _metadata_value(symbol: dict[str, Any], key: str, default: str) -> str:
    value = symbol.get(key)
    return value if isinstance(value, str) and value else default


def _write_batch(rows: list[dict[str, Any]], directory: Path, part: int) -> str:
    filename = f"part-{part:05d}.parquet"
    frame = pl.DataFrame(rows, schema=DETEXIFY_SCHEMA)
    frame.write_parquet(
        directory / filename,
        compression="zstd",
        statistics=True,
    )
    return filename


def _build_cache(
    sql_path: Path,
    symbols_path: Path,
    processed_dir: Path,
    *,
    batch_size: int = 5_000,
    validate_official_counts: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    symbols = _load_symbols(symbols_path)
    temporary = processed_dir.with_name(f".{processed_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    batch: list[dict[str, Any]] = []
    parts: list[str] = []
    class_counts: Counter[str] = Counter()
    raw_rows = 0
    output_rows = 0
    skipped_unlabeled = 0
    used_symbol_ids: set[str] = set()

    try:
        for sample_id, symbol_id, strokes in _iter_sql_samples(sql_path):
            raw_rows += 1
            if not symbol_id:
                skipped_unlabeled += 1
                continue
            symbol = symbols.get(symbol_id)
            if symbol is None:
                raise DatasetFormatError(
                    f"Sample {sample_id} references unknown symbol {symbol_id!r}"
                )
            command = symbol.get("command")
            if not isinstance(command, str) or not command:
                raise DatasetFormatError(f"Symbol {symbol_id!r} has no valid command")
            point_count = sum(len(stroke) for stroke in strokes)
            batch.append(
                {
                    "sample_id": sample_id,
                    "symbol_id": symbol_id,
                    "command": command,
                    "package": _metadata_value(symbol, "package", "latex2e"),
                    "font_encoding": _metadata_value(symbol, "fontenc", "OT1"),
                    "math_mode": bool(symbol.get("mathmode", False)),
                    "text_mode": bool(symbol.get("textmode", False)),
                    "strokes": strokes,
                    "stroke_count": len(strokes),
                    "point_count": point_count,
                }
            )
            output_rows += 1
            class_counts[symbol_id] += 1
            used_symbol_ids.add(symbol_id)
            if len(batch) >= batch_size:
                parts.append(_write_batch(batch, temporary, len(parts)))
                batch.clear()
                if progress is not None:
                    progress("convert", raw_rows, EXPECTED_RAW_ROWS)
        if batch:
            parts.append(_write_batch(batch, temporary, len(parts)))
            batch.clear()

        unused_symbols = set(symbols) - used_symbol_ids
        min_class_size = min(class_counts.values(), default=0)
        max_class_size = max(class_counts.values(), default=0)
        if validate_official_counts:
            expected = {
                "raw rows": (raw_rows, EXPECTED_RAW_ROWS),
                "output rows": (output_rows, EXPECTED_OUTPUT_ROWS),
                "unlabeled rows": (skipped_unlabeled, EXPECTED_UNLABELED_ROWS),
                "classes": (len(class_counts), EXPECTED_CLASSES),
                "minimum class size": (min_class_size, EXPECTED_MIN_CLASS_SIZE),
                "maximum class size": (max_class_size, EXPECTED_MAX_CLASS_SIZE),
                "unused symbols": (len(unused_symbols), 0),
            }
            mismatches = [
                f"{label}={actual} (expected {wanted})"
                for label, (actual, wanted) in expected.items()
                if actual != wanted
            ]
            if mismatches:
                raise DatasetIntegrityError(
                    "Official dataset validation failed: " + "; ".join(mismatches)
                )

        manifest: dict[str, Any] = {
            "schema_version": DATASET_VERSION,
            "schema": SCHEMA_DESCRIPTION,
            "sources": {
                SQL_DUMP.name: SQL_DUMP.sha256,
                SYMBOLS.name: SYMBOLS.sha256,
            },
            "parts": parts,
            "raw_rows": raw_rows,
            "output_rows": output_rows,
            "skipped_unlabeled_rows": skipped_unlabeled,
            "class_count": len(class_counts),
            "minimum_class_size": min_class_size,
            "maximum_class_size": max_class_size,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        temporary.replace(processed_dir)
        if progress is not None:
            progress("convert", raw_rows, raw_rows)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_valid_manifest(processed_dir: Path) -> dict[str, Any] | None:
    manifest_path = processed_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("schema_version") != DATASET_VERSION
        or manifest.get("schema") != SCHEMA_DESCRIPTION
        or manifest.get("output_rows") != EXPECTED_OUTPUT_ROWS
        or manifest.get("sources")
        != {
            SQL_DUMP.name: SQL_DUMP.sha256,
            SYMBOLS.name: SYMBOLS.sha256,
        }
        or not isinstance(manifest.get("parts"), list)
        or not manifest["parts"]
        or not all(
            isinstance(part, str) and (processed_dir / part).is_file()
            for part in manifest["parts"]
        )
    ):
        return None
    return manifest


def load_manifest(data_dir: str | Path = Path("data/detexify")) -> dict[str, Any]:
    """Return the validated cache manifest for an already imported dataset."""

    processed_dir = Path(data_dir) / CACHE_DIRECTORY
    manifest = _read_valid_manifest(processed_dir)
    if manifest is None:
        raise DatasetIntegrityError(
            f"No valid Detexify cache manifest exists at {processed_dir}"
        )
    return manifest


def load_detexify(
    data_dir: str | Path = Path("data/detexify"),
    *,
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Download, convert, cache, and return the official Detexify dataset.

    The first call downloads roughly 204 MiB and creates a partitioned Parquet
    cache. Later calls validate the manifest and read that cache directly.
    """

    data_dir = Path(data_dir)
    processed_dir = data_dir / CACHE_DIRECTORY
    manifest = _read_valid_manifest(processed_dir)
    if manifest is None:
        sources = _ensure_sources(data_dir, progress)
        manifest = _build_cache(
            sources[SQL_DUMP.name],
            sources[SYMBOLS.name],
            processed_dir,
            progress=progress,
        )

    part_paths = [processed_dir / part for part in manifest["parts"]]
    frame = pl.read_parquet(part_paths)
    if frame.height != manifest["output_rows"] or frame.schema != DETEXIFY_SCHEMA:
        raise DatasetIntegrityError(
            "The processed Detexify cache does not match its manifest"
        )
    return frame


def _point_coordinates(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        x, y = point.get("x"), point.get("y")
    elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
        if len(point) < 2:
            raise ValueError("Each point must contain x and y")
        x, y = point[0], point[1]
    else:
        raise TypeError("Each point must be a mapping or sequence")
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        raise ValueError("Point coordinates must be finite numbers")
    return float(x), float(y)


def rasterize_strokes(
    strokes: Sequence[Sequence[Any]],
    size: int = 64,
    *,
    margin: int = 6,
    stroke_width: int = 3,
    antialias_scale: int = 4,
) -> Image.Image:
    """Render native Detexify strokes as a centered grayscale Pillow image."""

    if size < 1 or margin < 0 or stroke_width < 1 or antialias_scale < 1:
        raise ValueError("Invalid raster dimensions")
    if margin * 2 >= size:
        raise ValueError("margin must leave a positive drawing area")

    parsed: list[list[tuple[float, float]]] = []
    for stroke in strokes:
        points = [_point_coordinates(point) for point in stroke]
        if points:
            parsed.append(points)
    if not parsed:
        raise ValueError("At least one point is required")

    all_points = [point for stroke in parsed for point in stroke]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    drawing_span = float(size - 2 * margin)
    scale_candidates = [
        drawing_span / extent for extent in (width, height) if extent > 0
    ]
    scale = min(scale_candidates) if scale_candidates else 1.0
    rendered_width = width * scale
    rendered_height = height * scale
    offset_x = (size - rendered_width) / 2.0 - min_x * scale
    offset_y = (size - rendered_height) / 2.0 - min_y * scale

    high_size = size * antialias_scale
    image = Image.new("L", (high_size, high_size), color=255)
    draw = ImageDraw.Draw(image)
    high_width = stroke_width * antialias_scale
    radius = high_width / 2.0

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] * scale + offset_x) * antialias_scale,
            (point[1] * scale + offset_y) * antialias_scale,
        )

    for stroke in parsed:
        coordinates = [transform(point) for point in stroke]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=0, width=high_width, joint="curve")
        for center in (coordinates[0], coordinates[-1]):
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                fill=0,
            )
    return image.resize((size, size), resample=Image.Resampling.LANCZOS)
