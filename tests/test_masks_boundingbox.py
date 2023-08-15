import numpy as np
import pytest

import tiger.masks


@pytest.fixture
def mask():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def bounding_box(mask):
    return tiger.masks.BoundingBox(mask)


@pytest.fixture
def empty_bounding_box(mask):
    return tiger.masks.BoundingBox(np.zeros_like(mask))


def test_bb_basic_properties(bounding_box):
    assert bounding_box.dims == 2
    assert tuple(bounding_box.lower_corner) == (1, 1)
    assert tuple(bounding_box.upper_corner) == (3, 7)
    assert bounding_box.shape == (3, 7)


def test_bb_empty(bounding_box, empty_bounding_box):
    assert not bounding_box.empty
    assert empty_bounding_box.empty


def test_bb_size(bounding_box, empty_bounding_box):
    assert bounding_box.size == 3 * 7
    assert len(bounding_box) == 3 * 7
    assert empty_bounding_box.size == 0


def test_bb_center(bounding_box, empty_bounding_box):
    assert tuple(bounding_box.center) == (2, 4)
    assert empty_bounding_box.center is None


def test_bb_contains(mask, bounding_box, empty_bounding_box):
    assert not bounding_box.contains([0, 0])
    assert [0, 0] not in bounding_box

    assert bounding_box.contains([1, 1])
    assert [1, 1] in bounding_box

    assert mask[2, 5] == 0 and bounding_box.contains([2, 5])
    assert not bounding_box.contains([4, 6])

    with pytest.raises(ValueError):
        bounding_box.contains([1, 2, 3])  # cannot test 3D coordinates with a 2D bounding box

    assert not empty_bounding_box.contains([0, 0])
    assert not empty_bounding_box.contains([1, 1])
    assert not empty_bounding_box.contains([2, 5])
    assert not empty_bounding_box.contains([4, 6])


def test_bb_mask(bounding_box, empty_bounding_box):
    expected_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert np.array_equal(bounding_box.make_mask(), expected_mask)
    assert np.array_equal(empty_bounding_box.make_mask(), np.zeros_like(expected_mask))


def test_bb_crop(bounding_box, empty_bounding_box):
    image = np.array(
        [
            [1, 6, 7, 4, 9, 4, 2, 5, 1, 9],
            [2, 5, 8, 3, 8, 5, 4, 3, 2, 8],
            [3, 4, 5, 6, 7, 6, 6, 8, 3, 7],
            [4, 3, 6, 5, 6, 7, 8, 6, 3, 7],
            [5, 2, 3, 8, 5, 8, 3, 4, 2, 8],
            [6, 1, 4, 7, 4, 9, 5, 2, 1, 9],
        ]
    )

    expected_patch = np.array([[5, 8, 3, 8, 5, 4, 3], [4, 5, 6, 7, 6, 6, 8], [3, 6, 5, 6, 7, 8, 6]])

    assert np.array_equal(bounding_box.crop(image), expected_patch)
    assert empty_bounding_box.crop(image).shape == (0, 0)

    with pytest.raises(ValueError):
        bounding_box.crop(np.zeros(image.shape + (2, 2)))


def test_bb_volume(mask):
    new_mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=-1)
    bounding_box = tiger.masks.BoundingBox(new_mask)

    assert bounding_box.dims == 3
    assert not bounding_box.empty
    assert tuple(bounding_box.lower_corner) == (1, 1, 1)
    assert tuple(bounding_box.upper_corner) == (3, 7, 1)
    assert bounding_box.size == 3 * 7
    assert tuple(bounding_box.center) == (2, 4, 1)

    assert bounding_box.contains([2, 5, 1])
    assert not bounding_box.contains([2, 5, 0])
