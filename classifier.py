"""Vision Transformer classifier for rasterized Detexify symbols."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

from detexify_data import rasterize_strokes


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings stored alongside every checkpoint."""

    image_size: int = 32
    patch_size: int = 4
    embedding_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.patch_size < 1 or self.image_size % self.patch_size:
            raise ValueError("patch_size must divide image_size exactly")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if self.embedding_dim < 1 or self.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if self.transformer_layers < 1 or self.feedforward_dim < 1:
            raise ValueError("transformer dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CommandVocabulary:
    """Stable mapping between class indices and LaTeX commands."""

    def __init__(self, commands: Iterable[str]) -> None:
        normalized = sorted(set(commands))
        if not normalized or any(not command for command in normalized):
            raise ValueError("At least one non-empty command is required")
        self.commands = tuple(normalized)
        self._indices = {command: index for index, command in enumerate(self.commands)}

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> CommandVocabulary:
        return cls(frame.get_column("command").unique().to_list())

    def __len__(self) -> int:
        return len(self.commands)

    def encode(self, command: str) -> int:
        try:
            return self._indices[command]
        except KeyError as error:
            raise KeyError(f"Unknown LaTeX command {command!r}") from error

    def decode(self, class_index: int) -> str:
        if class_index < 0 or class_index >= len(self.commands):
            raise IndexError(f"Invalid class index {class_index}")
        return self.commands[class_index]


class LatexCommandViT(nn.Module):
    """Vision Transformer that classifies a grayscale LaTeX symbol.

    The image is split into non-overlapping square patches. Each patch is
    projected directly into the Transformer embedding space, then a learned
    classification token is used to predict one LaTeX-command class.
    """

    def __init__(
        self,
        num_classes: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.config = config or ModelConfig()
        embedding_dim = self.config.embedding_dim

        # A convolution with kernel_size=stride=patch_size is exactly a shared
        # linear projection over non-overlapping flattened image patches.
        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        grid_size = self.config.image_size // self.config.patch_size
        token_count = grid_size * grid_size
        self.class_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, token_count + 1, embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(self.config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        expected_shape = (1, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected images shaped (batch, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}), got "
                f"{tuple(images.shape)}"
            )
        patches = self.patch_embedding(images)
        tokens = patches.flatten(start_dim=2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.transformer(tokens)
        return self.classifier(encoded[:, 0])


# Keep the descriptive classifier name as a backwards-compatible public API.
LatexCommandClassifier = LatexCommandViT
VisionTransformer = LatexCommandViT


def preprocess_image(
    image: Image.Image | np.ndarray[Any, Any],
    *,
    image_size: int = 32,
) -> Tensor:
    """Convert a grayscale raster to a normalized ``(1, H, W)`` tensor.

    The public raster convention is black ink on a white background. This
    function inverts it so the network receives background=0 and ink=1.
    """

    if isinstance(image, np.ndarray):
        values = image
        if values.ndim == 3 and values.shape[-1] in (3, 4):
            values = values[..., :3]
        if np.issubdtype(values.dtype, np.floating):
            maximum = float(np.nanmax(values)) if values.size else 0.0
            if maximum <= 1.0:
                values = values * 255.0
        values = np.clip(values, 0, 255).astype(np.uint8)
        image = Image.fromarray(values)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image or NumPy array")
    grayscale = image.convert("L")
    if grayscale.size != (image_size, image_size):
        grayscale = grayscale.resize(
            (image_size, image_size),
            resample=Image.Resampling.LANCZOS,
        )
    brightness = np.asarray(grayscale, dtype=np.float32)
    ink = np.ascontiguousarray((255.0 - brightness) / 255.0)
    return torch.from_numpy(ink).unsqueeze(0)


class DetexifyCommandDataset(Dataset[tuple[Tensor, Tensor]]):
    """Rasterize Detexify stroke records lazily for model training."""

    def __init__(
        self,
        frame: pl.DataFrame,
        vocabulary: CommandVocabulary | None = None,
        indices: Sequence[int] | np.ndarray[Any, Any] | None = None,
        *,
        image_size: int = 32,
    ) -> None:
        required = {"strokes", "command"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.frame = frame.select("strokes", "command")
        self.vocabulary = vocabulary or CommandVocabulary.from_frame(frame)
        self.image_size = image_size
        self.indices = (
            np.arange(frame.height, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= frame.height
        ):
            raise IndexError("A dataset index is outside the frame")
        self.targets = np.fromiter(
            (
                self.vocabulary.encode(command)
                for command in self.frame.get_column("command")
            ),
            dtype=np.int64,
            count=frame.height,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        source_index = int(self.indices[index])
        strokes = self.frame.row(source_index)[0]
        raster = rasterize_strokes(strokes, size=self.image_size)
        image = preprocess_image(raster, image_size=self.image_size)
        target = torch.tensor(self.targets[source_index], dtype=torch.long)
        return image, target


@dataclass(frozen=True)
class Prediction:
    command: str
    probability: float


@torch.inference_mode()
def predict_commands(
    model: LatexCommandClassifier,
    vocabulary: CommandVocabulary,
    image: Image.Image | np.ndarray[Any, Any],
    *,
    top_k: int = 5,
    device: torch.device | str | None = None,
) -> list[Prediction]:
    """Return the most likely LaTeX commands for one raster image."""

    if len(vocabulary) != model.num_classes:
        raise ValueError("Vocabulary size does not match the model head")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model_device = torch.device(device) if device else next(model.parameters()).device
    model.to(model_device)
    tensor = preprocess_image(image, image_size=model.config.image_size)
    was_training = model.training
    model.eval()
    probabilities = model(tensor.unsqueeze(0).to(model_device)).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(top_k, model.num_classes))
    if was_training:
        model.train()
    return [
        Prediction(vocabulary.decode(int(index)), float(score))
        for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
    ]


def save_classifier_checkpoint(
    path: str | Path,
    model: LatexCommandClassifier,
    vocabulary: CommandVocabulary,
    *,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Atomically save model weights and the exact command vocabulary."""

    if len(vocabulary) != model.num_classes:
        raise ValueError("Vocabulary size does not match the model head")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "metrics": metrics or {},
        "model_config": asdict(model.config),
        "commands": list(vocabulary.commands),
        "model_state": model.state_dict(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_classifier_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[LatexCommandClassifier, CommandVocabulary, dict[str, Any]]:
    """Restore a model, vocabulary, and training metadata."""

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported classifier checkpoint format")
    vocabulary = CommandVocabulary(payload["commands"])
    config = ModelConfig(**payload["model_config"])
    model = LatexCommandClassifier(len(vocabulary), config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    metadata = {
        "epoch": int(payload["epoch"]),
        "metrics": dict(payload.get("metrics", {})),
    }
    return model, vocabulary, metadata
