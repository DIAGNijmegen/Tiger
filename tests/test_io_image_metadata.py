import numpy as np
import pytest
import SimpleITK as sitk

from tiger.io import ImageMetadata


@pytest.fixture
def non_regular_direction():
    header = ImageMetadata(ndim=3)
    header["direction"] = np.array(
        [
            [0.9898719, -0.0951917, 0.1053199],
            [0.1053199, 0.9898719, -0.0951917],
            [-0.0951917, 0.1053199, 0.9898719],
        ]
    )
    return header


def test_has_default_direction(non_regular_direction):
    # 3D image
    header3 = ImageMetadata(ndim=3)
    assert header3.has_default_direction()

    header3["direction"] = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert header3.has_default_direction()

    header3["direction"] = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
    assert not header3.has_default_direction()

    header3["direction"] = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    assert not header3.has_default_direction()

    assert not non_regular_direction.has_default_direction()

    # 2D image
    header2 = ImageMetadata(ndim=2)
    assert header2.has_default_direction()

    header2["direction"] = (-1, 0, 0, 1)
    assert not header2.has_default_direction()


def test_has_regular_direction(non_regular_direction):
    header3 = ImageMetadata(ndim=3)
    assert header3.has_regular_direction()

    header3["direction"] = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert header3.has_regular_direction()

    header3["direction"] = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
    assert header3.has_regular_direction()

    header3["direction"] = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    assert header3.has_regular_direction()

    assert not non_regular_direction.has_regular_direction()


def generate_coordinate_transformation_example(ndim, use_sitk=False):
    if ndim == 2:
        spacing = (1, 0.8)
        origin = (124.86, -152.03)
        direction = (0, -1, 1, 0)

        physical_coordinates = (113.627, -152.779)
        indices = (-0.7489999999999952, 14.041250000000005)
    elif ndim == 3:
        # spacing = (0.7820000052452087, 0.7820000052452087, 0.800000011920929)
        # origin = (-193.94490000000002, -199.8043, 1358.0)
        # direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # physical_coordinates = (6.247099999999998, 76.2417, 1850.0)
        # indices = (255.99999828289842, 352.9999976322779, 614.999990835786)

        spacing = (0.4687499999999999, 0.4687500053964422, 0.8999999865193109)
        origin = (44.65591049194337, -131.92770385742188, 273.6060180664062)
        direction = (
            1.7100000072618052e-13,
            -0.03489943926091828,
            -0.9993908290249983,
            1.0,
            4.8969998498968526e-12,
            2.2186713214763487e-29,
            4.8940001357022906e-12,
            -0.9993908290249983,
            0.03489943926091828,
        )

        physical_coordinates = (33.81341236311808, -128.177703857415, 272.577538635301)
        indices = (8, 3, 12)
    else:
        raise ValueError(
            "Coordinate transformation examples are available only for 2D and 3D spaces"
        )

    if use_sitk:
        sitk_volume = sitk.Image([1] * ndim, sitk.sitkInt16)
        sitk_volume.SetSpacing(spacing)
        sitk_volume.SetOrigin(origin)
        sitk_volume.SetDirection(direction)

        indices = sitk_volume.TransformPhysicalPointToContinuousIndex(physical_coordinates)

    header = ImageMetadata(ndim, spacing=spacing, origin=origin, direction=direction)
    return header, physical_coordinates, indices


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("compare_with_sitk", (False, True))
def test_physical_coordinates_to_indices(ndim, compare_with_sitk):
    # Translate physical coordinates to indices
    header, physical_coordinates, indices = generate_coordinate_transformation_example(
        ndim, compare_with_sitk
    )

    calculated_indices_cont = header.physical_coordinates_to_indices(
        physical_coordinates, continuous=True
    )
    assert np.all(np.isclose(calculated_indices_cont, indices))

    calculated_indices = header.physical_coordinates_to_indices(
        physical_coordinates, continuous=False
    )
    assert np.all(np.isclose(calculated_indices, np.around(indices)))


@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("compare_with_sitk", (False, True))
def test_indices_to_physical_coordinates(ndim, compare_with_sitk):
    # Translate indices to physical coordinates
    header, physical_coordinates, indices = generate_coordinate_transformation_example(
        ndim, compare_with_sitk
    )

    calculated_physical_cont = header.indices_to_physical_coordinates(indices)
    assert np.all(np.isclose(calculated_physical_cont, physical_coordinates))


def test_strip_metadata():
    header = ImageMetadata(ndim=3)
    header["foo"] = 123
    assert "foo" in header
    header.strip()
    assert "foo" not in header
    assert "spacing" in header
