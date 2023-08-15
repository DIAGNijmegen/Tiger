from typing import Tuple

import numpy as np
import pytest
from pytest import approx

import tiger.metrics
from tiger.io import ImageMetadata


def test_bland_altman(random_state: np.random.RandomState):
    a = random_state.randint(low=-100, high=101, size=1000000)
    b = a + random_state.normal(loc=-2.5, scale=30, size=len(a))
    ba = tiger.metrics.BlandAltmanLimitsOfAgreement(a, b)
    assert ba.mean_diff == approx(2.5, abs=0.01)
    assert ba.lower_limit == approx(2.5 - 1.96 * 30, abs=0.1)
    assert ba.upper_limit == approx(2.5 + 1.96 * 30, abs=0.1)
    assert len(ba.means) == len(a)
    assert len(ba.diffs) == len(a)


def test_jaccard_score():
    shape = (24, 24, 24)

    # Input and output shapes have to match
    with pytest.raises(ValueError):
        tiger.metrics.jaccard_score(np.ones(shape), np.ones(shape + (2, 2, 2)))

    # Cannot compute Jaccard index for an empty mask (=no foreground)
    with pytest.raises(ValueError):
        tiger.metrics.jaccard_score(np.zeros(shape), np.zeros(shape))

    # Everything foreground -> perfect overlap
    assert tiger.metrics.jaccard_score(np.ones(shape), np.ones(shape)) == approx(1)
    assert tiger.metrics.jaccard_score(np.ones(shape), np.zeros(shape)) == approx(0)
    assert tiger.metrics.jaccard_score(np.zeros(shape), np.ones(shape)) == approx(0)

    half_mask = np.zeros(shape)
    half_mask[:12, :, :] = 1
    assert tiger.metrics.jaccard_score(half_mask, np.ones(shape)) == approx(0.5)
    assert tiger.metrics.jaccard_score(half_mask, np.zeros(shape)) == approx(0)

    eighth_mask = np.zeros(shape)
    eighth_mask[:12, :12, :12] = 1
    assert tiger.metrics.jaccard_score(eighth_mask, np.ones(shape)) == approx(0.125)


def test_dice_score():
    shape = (24, 24, 24)

    # Input and output shapes have to match
    with pytest.raises(ValueError):
        tiger.metrics.dice_score(np.ones(shape), np.ones(shape + (2, 2, 2)))

    # Cannot compute Jaccard index for an empty mask (=no foreground)
    with pytest.raises(ValueError):
        tiger.metrics.dice_score(np.zeros(shape), np.zeros(shape))

    # Everything foreground -> perfect overlap
    assert tiger.metrics.dice_score(np.ones(shape), np.ones(shape)) == approx(1)
    assert tiger.metrics.dice_score(np.ones(shape), np.zeros(shape)) == approx(0)
    assert tiger.metrics.dice_score(np.zeros(shape), np.ones(shape)) == approx(0)

    half_mask = np.zeros(shape)
    half_mask[:12, :, :] = 1
    assert tiger.metrics.dice_score(half_mask, np.ones(shape)) == approx(2 / 3)
    assert tiger.metrics.dice_score(half_mask, np.zeros(shape)) == approx(0)

    eighth_mask = np.zeros(shape)
    eighth_mask[:12, :12, :12] = 1
    assert tiger.metrics.dice_score(eighth_mask, np.ones(shape)) == approx(2 / 9)


def test_mean_dice_score():
    shape = (24, 24, 24)

    mask = np.zeros(shape)
    mask[:12, :12, :12] = 1
    mask[-6:, -6:, -6:] = 2

    ref_mask = np.zeros(shape)
    ref_mask[:16, :4, :12] = 1
    ref_mask[-10:, :, -4:] = 2

    dice_scores = {i: tiger.metrics.dice_score(mask == i, ref_mask == i) for i in (1, 2)}

    # Mean dice is mean of the individual dice scores
    mean_dice = sum(dice_scores.values()) / 2
    assert tiger.metrics.mean_dice_score(mask, ref_mask, labels=[1, 2]) == approx(mean_dice)

    # Labels can be ignored
    assert tiger.metrics.mean_dice_score(mask, ref_mask, labels=[2]) == approx(dice_scores[2])

    # Cannot compute dice of empty mask
    with pytest.raises(ValueError):
        tiger.metrics.mean_dice_score(np.zeros(shape), np.zeros(shape), labels=[])
    with pytest.raises(ValueError):
        tiger.metrics.mean_dice_score(np.zeros(shape), np.zeros(shape), labels=[1, 2])
    with pytest.raises(ValueError):
        tiger.metrics.mean_dice_score(np.ones(shape), np.ones(shape), labels=[1, 2])

    # Empty masks = dice score is 0
    assert tiger.metrics.mean_dice_score(np.ones(shape), np.zeros(shape), labels=[1]) == approx(0)
    assert tiger.metrics.mean_dice_score(np.zeros(shape), np.ones(shape), labels=[1]) == approx(0)


@pytest.fixture()
def calcium_image_and_mask() -> Tuple[np.ndarray, np.ndarray, ImageMetadata]:
    image = np.full(shape=(20, 20, 20), fill_value=129, dtype="int16")
    image[10:14, 8:12, 6:8] = 130
    image[10, 8, 6:8] = 200
    image[11, 8, 6:8] = 300
    image[12, 8, 6:8] = 400
    image[14:15, 12:14, 8:14] = 500

    mask = np.zeros(image.shape, dtype="uint8")
    mask[10:14, 8:11, 7:8] = 1
    mask[14:15, 12:13, 8:10] = 1

    header = ImageMetadata(ndim=3, spacing=[0.75, 0.75, 2.0], slice_thickness=2.5)
    return image, mask, header


def test_agatston_score(calcium_image_and_mask):
    image, mask, header = calcium_image_and_mask

    # Image and mask need to have the same shape
    with pytest.raises(ValueError):
        tiger.metrics.agatston_score(image, mask[:, :, 0], header["spacing"])

    assert tiger.metrics.agatston_score(image, mask, header["spacing"]) == 21


def test_volume_score(calcium_image_and_mask):
    _, mask, header = calcium_image_and_mask
    assert tiger.metrics.volume_score(mask, header["spacing"]) == approx(15.75)


def test_calcium_mass_score(calcium_image_and_mask):
    image, mask, header = calcium_image_and_mask

    # Image and mask need to have the same shape
    with pytest.raises(ValueError):
        tiger.metrics.mass_score(image, mask[:, :, 0], header["spacing"])

    assert tiger.metrics.mass_score(image, mask, header["spacing"]) == approx(3.45375)
