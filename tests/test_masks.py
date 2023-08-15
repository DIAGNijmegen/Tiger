"""Module defining tests for the merging of the masks as defined in masks.py"""

import numpy as np
import pytest

from tiger.io import ImageMetadata
from tiger.masks import OverlappingMasksError, merge_masks, most_common_labels


def test_most_common_labels():
    assert most_common_labels([0, 1, 1, 2, 2, 2, 1, 1]) == [1, 2]
    assert most_common_labels([0, 1, 1, 2, 2, 2, 1, 1], background=1) == [2, 0]
    assert most_common_labels([0, 0, 0]) == []
    assert most_common_labels([]) == []


def test_merge_masks_3d():
    """Test for merging multiple masks"""  # TODO: split up tests into multiple smaller functions
    ref_shape = (512, 512, 256)
    ref_header = ImageMetadata(ndim=3, origin=(-150, -150, 150))

    # Spacing of the masks should not make a difference.
    mask1 = (np.ones((20, 16, 32)), ImageMetadata(ndim=3, origin=(50, 50, 150)))
    mask2 = (np.ones((40, 40, 14)), ImageMetadata(ndim=3, origin=(100, 100, 250)))
    mask2[0][0, :, :] = 0  # make sure there is some background in the mask

    # One row (1st dimension overlaps with mask2).
    mask3 = (np.ones((20, 16, 32)), ImageMetadata(ndim=3, origin=(138, 139, 263)))

    # Out of bounds mask.
    mask4 = (np.ones((50, 50, 50)), ImageMetadata(ndim=3, origin=(-160, 340, 250)))

    # Different spacing
    mask5 = (mask1[0].copy(), mask1[1].copy())
    mask5[1]["spacing"] = (2, 2, 4)

    non_overlapping_masks = [mask1, mask2]
    overlapping_masks = [mask2, mask3]

    mask = merge_masks(
        reference_image_shape=(512, 512, 256),
        reference_image_header=ref_header,
        masks=non_overlapping_masks,
        unique_labels=True,
        strict=True,
    )

    # Test if output is 3d.
    assert mask.ndim == 3

    # Test whether the masks are set to a label in the mask at the expected
    # location.
    assert np.all(mask[200:220, 200:216, 0:32] == 1)  # for mask1
    assert np.all(mask[251:290, 250:290, 100:114] == 2)  # for mask2
    assert np.all(mask[250, 250:290, 100:114] == 0)  # background portion of mask 2

    # Test unique_labels parameter - binary mask is expected.
    mask = merge_masks(
        reference_image_shape=ref_shape,
        reference_image_header=ref_header,
        masks=non_overlapping_masks,
        unique_labels=False,
        strict=True,
    )
    assert np.all((mask == 0) | (mask == 1))

    # Test binarization
    non_overlapping_non_binary_masks = [(mask1[0] * 5, mask1[1]), mask2]

    mask = merge_masks(
        reference_image_shape=ref_shape,
        reference_image_header=ref_header,
        masks=non_overlapping_non_binary_masks,
        unique_labels=False,
        binarize=False,
        strict=True,
    )
    assert np.all((mask == 0) | (mask == 1) | (mask == 5))

    mask = merge_masks(
        reference_image_shape=ref_shape,
        reference_image_header=ref_header,
        masks=non_overlapping_non_binary_masks,
        unique_labels=False,
        binarize=True,
        strict=True,
    )
    assert np.all((mask == 0) | (mask == 1))

    mask = merge_masks(
        reference_image_shape=ref_shape,
        reference_image_header=ref_header,
        masks=non_overlapping_non_binary_masks,
        unique_labels=True,
        binarize=True,
        strict=True,
    )
    assert np.all((mask == 0) | (mask == 1) | (mask == 2))

    # Test if the exception is thrown when masks overlap and strict mode is on.
    with pytest.raises(OverlappingMasksError):
        merge_masks(
            reference_image_shape=ref_shape,
            reference_image_header=ref_header,
            masks=overlapping_masks,
            unique_labels=True,
            strict=True,
        )

    # Test if the exception is NOT thrown when masks overlap but strict mode is off.
    merge_masks(
        reference_image_shape=ref_shape,
        reference_image_header=ref_header,
        masks=overlapping_masks,
        unique_labels=True,
        strict=False,
    )

    # Test by simply creating this, as it can throw an incompatible shape error
    # as it should successfully crop before merging.
    mask = merge_masks(
        reference_image_shape=(500, 500, 250),
        reference_image_header=ref_header,
        masks=[mask4],
        unique_labels=False,
        strict=False,
    )

    # Test if the mask is indeed cropped and at its expected location.
    assert np.all(mask[:40, 490:, 100:150])

    # Different spacing should not be tolerated
    with pytest.raises(ValueError):
        merge_masks(
            reference_image_shape=ref_shape,
            reference_image_header=ref_header,
            masks=[mask1, mask2, mask3, mask5],
        )


def test_merge_masks_2d():
    # Test for 2d images.
    ref_header = ImageMetadata(ndim=2, origin=(-150, -150))

    mask1 = (np.ones((20, 16)), ImageMetadata(ndim=2, origin=(50, 50)))
    mask2 = (np.ones((40, 40)), ImageMetadata(ndim=2, origin=(100, 100)))

    masks = [mask1, mask2]

    # Just assert that it returns a merged mask.
    # Parameter settings have been checked for 3d images and this is to check
    # whether it is generalizable to all dimensions.
    mask = merge_masks(
        reference_image_shape=(512, 512),
        reference_image_header=ref_header,
        masks=masks,
        unique_labels=True,
        strict=True,
    )
    assert isinstance(mask, np.ndarray)
