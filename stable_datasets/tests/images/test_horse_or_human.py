import pytest
from PIL import Image

from stable_datasets.images.horse_or_human import HorseOrHuman


@pytest.fixture(scope="module")
def dataset():
    # Download and prepare the dataset once for all tests
    return HorseOrHuman()


def test_dataset_info():
    """Check dataset metadata."""
    from types import SimpleNamespace

    builder = SimpleNamespace(SOURCE=HorseOrHuman.SOURCE)
    info = HorseOrHuman._info(builder)

    assert info is not None
    assert info.supervised_keys == ("image", "label")
    assert len(info.features["label"].names) == 2


def test_train_split(dataset):
    """Check training split size."""
    train = dataset["train"]

    assert len(train) == 1027


def test_test_split(dataset):
    """Check test split size."""
    test = dataset["test"]

    assert len(test) == 256


@pytest.mark.parametrize("split", ["train", "test"])
def test_sample_structure(dataset, split):
    """Check sample fields and values."""
    sample = dataset[split][0]

    assert "image" in sample
    assert "label" in sample
    assert int(sample["label"]) in (0, 1)


@pytest.mark.parametrize("split", ["train", "test"])
def test_all_labels(dataset, split):
    """Check every label is valid and both classes exist."""
    samples = dataset[split]

    labels = [int(sample["label"]) for sample in samples]

    assert all(label in (0, 1) for label in labels)
    assert set(labels) == {0, 1}


@pytest.mark.parametrize("split", ["train", "test"])
def test_image_properties(dataset, split):
    """Check image type, RGB format, and dimensions."""
    samples = dataset[split]

    # Inspect a few images from each split
    for index in [0, len(samples) // 2, len(samples) - 1]:
        image = samples[index]["image"]

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (300, 300)


@pytest.mark.parametrize("split", ["train", "test"])
def test_unique_samples(dataset, split):
    """Check there are no duplicate sample images."""
    samples = dataset[split]

    # Compare image contents, not object identities
    hashes = [
        hash(sample["image"].tobytes())
        for sample in samples
    ]

    assert len(hashes) == len(set(hashes))


def test_invalid_split():
    """Check unsupported split names are rejected."""
    with pytest.raises(ValueError, match="Unknown split"):
        list(
            HorseOrHuman._generate_examples(
                None, "unused.zip", "invalid"
            )
        )