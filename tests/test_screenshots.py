import imageio
import numpy as np
import pytest

import tiger.screenshots

approx = pytest.approx


@pytest.mark.parametrize(
    "lut_class",
    [
        tiger.screenshots.DefaultColors,
        tiger.screenshots.VertebraColors,
        tiger.screenshots.SpineColors,
    ],
)
def test_default_color_lut(lut_class):
    lut = lut_class()

    # Color for label 0 should be transparent
    assert lut(0) == (0, 0, 0, 0)

    # Colors should be 4 numbers, between 0 and 255, and not the same
    c1 = lut(1)
    assert len(c1) == 4
    assert all(0 <= c <= 255 for c in c1)

    # All colors in these LUTs should be unique
    colors = set(lut)
    assert len(lut) > 1
    assert len(colors) == len(lut)


@pytest.mark.parametrize(
    "lut_class", [tiger.screenshots.VertebraColors, tiger.screenshots.SpineColors]
)
def test_vertebra_color_lut(lut_class):
    opacity = 0.5
    lut = lut_class(incomplete_opacity=opacity)

    # Color for label 101 should be the same as for 1, just different transparency
    for label in range(1, 26):
        c1 = lut(label)
        c101 = lut(100 + label)
        assert c1[:3] == c101[:3]
        assert round(c1[3] * opacity) == approx(c101[3])


def test_spine_color_lut():
    opacity = 0.75
    lut = tiger.screenshots.SpineColors(disc_opacity=opacity)

    # Color for label 100 should have full opactiy
    assert lut(100)[3] == 255

    # Colors for labels 201+ should have a specific transparency
    for label in range(201, 226):
        assert lut(label)[3] == round(255 * opacity)


def test_rib_color_lut():
    lut = tiger.screenshots.RibColors()

    # Color for label 0 should be transparent
    assert lut(0) == (0, 0, 0, 0)

    # Color for label 1 should be red
    assert lut(1) == (255, 0, 0, 255)

    # Color for label 101 should be the same as for 201
    for label in range(101, 114):
        c = lut(label)
        assert c[3] > 0
        assert c == lut(label + 100)


def test_screenshot_generator(resources_path, mri_image_iso):
    image, header = mri_image_iso

    # Simple axial screenshot
    generator = tiger.screenshots.ScreenshotGenerator(coordinate=0.25, axis=2)
    assert np.array_equal(
        generator(image), imageio.imread(resources_path / "mri_screenshot_iso_axial.png")
    )

    # Simple sagittal screenshot
    generator = tiger.screenshots.ScreenshotGenerator(coordinate=0.25, axis=0)
    assert np.array_equal(
        generator(image), imageio.imread(resources_path / "mri_screenshot_iso_sagittal.png")
    )
