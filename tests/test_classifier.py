from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

from classifier import (
    CommandVocabulary,
    DetexifyCommandDataset,
    LatexCommandClassifier,
    ModelConfig,
    load_classifier_checkpoint,
    predict_commands,
    preprocess_image,
    save_classifier_checkpoint,
)
from train_classifier import stratified_split_indices

SMALL_CONFIG = ModelConfig(
    embedding_dim=32,
    attention_heads=4,
    transformer_layers=1,
    feedforward_dim=64,
    dropout=0.0,
)


def test_model_maps_32_pixel_images_to_class_logits_and_backpropagates() -> None:
    model = LatexCommandClassifier(num_classes=7, config=SMALL_CONFIG)
    images = torch.rand(3, 1, 32, 32)

    logits = model(images)
    logits.mean().backward()

    assert logits.shape == (3, 7)
    assert model.patch_embedding.kernel_size == (4, 4)
    assert model.patch_embedding.stride == (4, 4)
    assert model.position_embedding.shape == (1, 65, 32)
    assert model.class_token.grad is not None
    assert model.patch_embedding.weight.grad is not None
    with pytest.raises(ValueError, match="Expected images"):
        model(torch.rand(3, 1, 64, 64))


def test_model_config_requires_non_overlapping_patches_to_tile_image() -> None:
    with pytest.raises(ValueError, match="patch_size must divide image_size"):
        ModelConfig(image_size=32, patch_size=5)


def test_vocabulary_is_sorted_unique_and_decodes_commands() -> None:
    vocabulary = CommandVocabulary([r"\sum", r"\alpha", r"\sum"])

    assert vocabulary.commands == (r"\alpha", r"\sum")
    assert vocabulary.encode(r"\sum") == 1
    assert vocabulary.decode(0) == r"\alpha"
    with pytest.raises(KeyError, match="Unknown"):
        vocabulary.encode(r"\unknown")
    with pytest.raises(IndexError, match="Invalid"):
        vocabulary.decode(-1)


def test_preprocess_image_inverts_black_ink_and_resizes() -> None:
    raster = np.full((16, 16), 255, dtype=np.uint8)
    raster[8, 8] = 0

    tensor = preprocess_image(raster)
    white = preprocess_image(Image.new("L", (32, 32), 255))

    assert tensor.shape == (1, 32, 32)
    assert tensor.dtype == torch.float32
    assert 0 <= float(tensor.min()) <= float(tensor.max()) <= 1
    assert float(tensor.max()) > 0.5
    assert torch.count_nonzero(white) == 0


def test_dataset_rasterizes_strokes_and_encodes_command() -> None:
    frame = pl.DataFrame(
        {
            "command": [r"\beta", r"\alpha"],
            "strokes": [
                [[[0, 0, 1], [10, 20, 2]]],
                [[[5, 0, 1], [5, 20, 2]]],
            ],
        }
    )
    vocabulary = CommandVocabulary.from_frame(frame)
    dataset = DetexifyCommandDataset(frame, vocabulary)

    image, target = dataset[1]

    assert len(dataset) == 2
    assert image.shape == (1, 32, 32)
    assert float(image.max()) > 0.5
    assert target.dtype == torch.long
    assert vocabulary.decode(int(target)) == r"\alpha"


def test_stratified_split_retains_each_class_for_training() -> None:
    targets = np.array([0] * 4 + [1] * 10 + [2] * 2, dtype=np.int64)

    train, validation = stratified_split_indices(
        targets, validation_fraction=0.25, seed=7
    )

    assert set(train).isdisjoint(validation)
    assert sorted(np.concatenate((train, validation)).tolist()) == list(range(16))
    assert set(targets[train]) == {0, 1, 2}
    assert np.bincount(targets[validation], minlength=3).tolist() == [1, 2, 1]
    repeated_train, repeated_validation = stratified_split_indices(
        targets, validation_fraction=0.25, seed=7
    )
    assert np.array_equal(train, repeated_train)
    assert np.array_equal(validation, repeated_validation)


def test_prediction_returns_ranked_latex_commands() -> None:
    vocabulary = CommandVocabulary([r"\alpha", r"\beta", r"\gamma"])
    model = LatexCommandClassifier(len(vocabulary), SMALL_CONFIG)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([0.0, 3.0, 1.0]))

    predictions = predict_commands(
        model,
        vocabulary,
        Image.new("L", (32, 32), 255),
        top_k=2,
    )

    assert [prediction.command for prediction in predictions] == [
        r"\beta",
        r"\gamma",
    ]
    assert predictions[0].probability > predictions[1].probability


def test_checkpoint_round_trip_preserves_weights_and_vocabulary(
    tmp_path: Path,
) -> None:
    torch.manual_seed(3)
    vocabulary = CommandVocabulary([r"\alpha", r"\beta", r"\gamma"])
    model = LatexCommandClassifier(len(vocabulary), SMALL_CONFIG).eval()
    images = torch.rand(2, 1, 32, 32)
    expected_logits = model(images)
    checkpoint = tmp_path / "classifier.pt"

    save_classifier_checkpoint(
        checkpoint,
        model,
        vocabulary,
        epoch=4,
        metrics={"validation_top1_accuracy": 0.75},
    )
    restored, restored_vocabulary, metadata = load_classifier_checkpoint(checkpoint)
    restored.eval()

    assert restored_vocabulary.commands == vocabulary.commands
    assert metadata == {
        "epoch": 4,
        "metrics": {"validation_top1_accuracy": 0.75},
    }
    assert torch.equal(expected_logits, restored(images))
