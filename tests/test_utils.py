import numpy as np
import pytest

import tiger.utils


def test_repeat_if_needed():
    # Single number becomes tuple of n-times that number
    assert tiger.utils.repeat_if_needed(5, 3) == (5, 5, 5)

    # List of n items becomes tuple of n items
    t = tiger.utils.repeat_if_needed([1, 2, 3], 3)
    assert t == (1, 2, 3)
    assert isinstance(t, tuple)

    # List of m items raises an error
    with pytest.raises(ValueError):
        tiger.utils.repeat_if_needed([1, 2], 3)

    # Returnd tuple has the specified type
    assert all(isinstance(i, int) for i in tiger.utils.repeat_if_needed(5, 3))
    assert all(isinstance(i, float) for i in tiger.utils.repeat_if_needed(5.1, 3))

    # None is not an accepted input
    with pytest.raises(ValueError):
        tiger.utils.repeat_if_needed(None, 3)


def test_slice_nd_array():
    nd_array = np.arange(64**3).reshape((64, 64, 64))

    slice_without_offset = nd_array[tiger.utils.slice_nd_array(slice_shape=(12, 32, 16))]
    assert np.array_equal(slice_without_offset, nd_array[:12, :32, :16])

    slice_with_offset = nd_array[
        tiger.utils.slice_nd_array(slice_shape=(12, 32, 16), offset=(4, 2, 0))
    ]
    assert np.array_equal(slice_with_offset, nd_array[4:16, 2:34, 0:16])


def test_first():
    assert tiger.utils.first([1, 2, 3]) == 1
    assert tiger.utils.first([1, 2, 3], lambda x: x > 1) == 2


def test_count():
    assert tiger.utils.count(i in (2, 3) for i in range(5)) == 2
    assert tiger.utils.count([1, 2, 4, 5, 8], lambda x: x > 2) == 3
