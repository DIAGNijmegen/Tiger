import numpy as np
import pytest

import tiger.masks


@pytest.fixture
def binary_mask():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def multi_label_mask():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 2, 2, 2, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


def test_rlc_connectivity(binary_mask):
    # Default connectivity is full connectivity
    assert np.array_equal(
        tiger.masks.retain_largest_components(binary_mask),
        tiger.masks.retain_largest_components(binary_mask, connectivity=2),
    )

    assert np.array_equal(
        tiger.masks.retain_largest_components(binary_mask, connectivity=1),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )

    assert np.array_equal(
        tiger.masks.retain_largest_components(binary_mask, connectivity=2),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_rlc_multiple_components(binary_mask):
    # Retaining more than one component
    assert np.array_equal(
        tiger.masks.retain_largest_components(binary_mask, n=2),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert np.array_equal(tiger.masks.retain_largest_components(binary_mask, n=3), binary_mask)
    assert np.array_equal(tiger.masks.retain_largest_components(binary_mask, n=4), binary_mask)
    assert np.array_equal(
        tiger.masks.retain_largest_components(binary_mask, n=0),
        np.zeros_like(binary_mask),
    )


def test_rlc_multiple_labels(multi_label_mask):
    # Multiple objects
    assert np.array_equal(
        tiger.masks.retain_largest_components(multi_label_mask),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_rlc_specify_foreground(multi_label_mask):
    # Choosing relevant foreground labels manually
    assert np.array_equal(
        tiger.masks.retain_largest_components(multi_label_mask, labels=[2]),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_rlc_different_background(binary_mask):
    # Change background from 0 to -1
    binary_mask_bg = binary_mask.copy()
    binary_mask_bg[binary_mask == 0] = -1

    # Remove small components
    largest_component = tiger.masks.retain_largest_components(binary_mask_bg, background=-1)
    assert tuple(np.unique(largest_component)) == (-1, 1)

    # Put original background back, then it should be the same mask
    largest_component[largest_component == -1] = 0
    assert np.array_equal(largest_component, tiger.masks.retain_largest_components(binary_mask))


def test_cc_binary(binary_mask):
    cc = tiger.masks.ConnectedComponents(binary_mask)
    assert len(cc) == 3

    assert cc[0].size == 1
    assert cc[1].size == 2
    assert cc[2].size == 6

    assert cc.labels() == [1]
    for c in cc:
        assert c.label == 1


def test_cc_multiple_labels(multi_label_mask):
    cc = tiger.masks.ConnectedComponents(multi_label_mask)
    assert len(cc) == 5
    assert cc.labels() == [1, 2]

    sizes = (1, 1, 2, 3, 6)
    labels = (1, 2, 1, 2, 1)

    for c, size, label in zip(cc, sizes, labels):
        assert c.size == size
        assert c.label == label
