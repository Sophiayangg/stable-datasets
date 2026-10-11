import numpy as np
from PIL import Image

from stable_datasets.images import Semeion


def test_semeion_splits():
    """Check that both splits load with correct sizes."""
    train = Semeion(split="train")
    test = Semeion(split="test")

    assert len(train) == 1274
    assert len(test) == 319
    assert len(train) + len(test) == 1593


def test_semeion_sample_structure():
    """Check the structure of a dataset example."""
    train = Semeion(split="train")
    sample = train[0]

    assert set(sample.keys()) == {"image", "label"}
    assert isinstance(sample["image"], Image.Image)
    assert isinstance(sample["label"], int)


def test_semeion_image():
    """Check image dimensions, mode, and pixel values."""
    train = Semeion(split="train")
    image = train[0]["image"]

    assert image.size == (16, 16)
    assert image.mode == "L"

    pixels = np.asarray(image)

    assert pixels.shape == (16, 16)
    assert pixels.dtype == np.uint8
    assert np.isin(pixels, [0, 255]).all()


def test_semeion_labels():
    """Check that all examples have valid digit labels."""
    train = Semeion(split="train")
    test = Semeion(split="test")

    for dataset in [train, test]:
        for sample in dataset:
            assert isinstance(sample["label"], int)
            assert 0 <= sample["label"] <= 9


def test_semeion_deterministic():
    """Check that repeated loads return the same examples."""
    train1 = Semeion(split="train")
    train2 = Semeion(split="train")

    assert len(train1) == len(train2)

    for idx in [0, 10, 100, 500]:
        sample1 = train1[idx]
        sample2 = train2[idx]

        assert sample1["label"] == sample2["label"]
        assert np.array_equal(
            np.asarray(sample1["image"]),
            np.asarray(sample2["image"]),
        )


def test_semeion_split_indices():
    """Check that the configured split indices do not overlap."""
    indices = np.random.default_rng(42).permutation(1593)

    train_indices = set(indices[:1274])
    test_indices = set(indices[1274:])

    assert train_indices.isdisjoint(test_indices)
    assert len(train_indices | test_indices) == 1593
