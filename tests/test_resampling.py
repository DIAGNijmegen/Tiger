import numpy as np
import pytest

import tiger.io
import tiger.resampling


def test_resample_image(mri_image, mri_image_iso):
    # Resample image to isotropic resolution and compare with reference image
    image, header = mri_image
    image_iso, header_iso = mri_image_iso

    image_resampled = tiger.resampling.resample_image(
        image, header["spacing"], header_iso["spacing"]
    )
    assert np.array_equal(image_resampled, image_iso)


def test_resample_image_itk(ct_image, ct_image_iso):
    # Resample image to isotropic resolution and compare with reference image
    image, header = ct_image
    image_iso, header_iso = ct_image_iso

    image_resampled = tiger.resampling.resample_image_itk(
        image, header["spacing"], header_iso["spacing"], interpolation="lanczos"
    )
    assert np.array_equal(image_resampled, image_iso)


def test_resample_mask(mri_mask, mri_mask_iso):
    # Resample mask to isotropic resolution and compare with reference mask
    mask, header = mri_mask
    mask_iso, header_iso = mri_mask_iso

    mask_resampled = tiger.resampling.resample_mask(mask, header["spacing"], header_iso["spacing"])
    assert np.array_equal(mask_resampled, mask_iso)


def test_resample_mask_itk(ct_mask, ct_mask_iso):
    # Resample mask to isotropic resolution and compare with reference mask
    mask, header = ct_mask
    mask_iso, header_iso = ct_mask_iso

    mask_resampled = tiger.resampling.resample_mask_itk(
        mask, header["spacing"], header_iso["spacing"]
    )
    assert np.array_equal(mask_resampled, mask_iso)


def test_resample_mask_distance_transform(ct_mask, ct_mask_iso_dt):
    # Resample mask to isotropic resolution and compare with reference mask
    mask, header = ct_mask
    mask_iso, header_iso = ct_mask_iso_dt

    mask_resampled = tiger.resampling.resample_mask_dt(
        mask, header["spacing"], header_iso["spacing"]
    )
    assert np.array_equal(mask_resampled, mask_iso)


def test_crop(mri_image):
    image, header = mri_image
    assert image.shape == (39, 305, 305)

    image_cropped = tiger.resampling.pad_or_crop_image(
        image, target_shape=(12, 12, 12), align="min"
    )
    assert np.array_equal(image_cropped, image[:12, :12, :12])

    image_cropped = tiger.resampling.pad_or_crop_image(
        image, target_shape=(12, 12, 12), align="max"
    )
    assert np.array_equal(image_cropped, image[-12:, -12:, -12:])

    image_cropped = tiger.resampling.pad_or_crop_image(
        image, target_shape=(12, 12, 12), align="center"
    )
    assert np.array_equal(image_cropped, image[13:25, 146:158, 146:158])

    image_cropped = tiger.resampling.pad_or_crop_image(
        image, target_shape=(12, 14, 16), align=("min", "center", "max")
    )
    assert np.array_equal(image_cropped, image[:12, 145:159, -16:])


def test_pad(mri_image):
    image, header = mri_image
    assert image.shape == (39, 305, 305)

    fill_value = 100
    shape = (52, 400, 411)

    image_padded = tiger.resampling.pad_or_crop_image(
        image, target_shape=shape, fill=fill_value, align="min"
    )
    assert image_padded.shape == shape
    assert np.array_equal(image_padded[:39, :305, :305], image)
    assert np.all(image_padded[39:, 305:, 305:] == fill_value)

    image_padded = tiger.resampling.pad_or_crop_image(
        image, target_shape=shape, fill=fill_value, align="max"
    )
    assert image_padded.shape == shape
    assert np.array_equal(image_padded[-39:, -305:, -305:], image)
    assert np.all(image_padded[:-39, :-305, :-305] == fill_value)

    image_padded = tiger.resampling.pad_or_crop_image(
        image, target_shape=shape, fill=fill_value, align="center"
    )
    assert image_padded.shape == shape
    image_center = image_padded[6:-7, 47:-48, 53:-53]
    assert np.array_equal(image_center, image)
    image_center.fill(fill_value)
    assert np.all(image_padded == fill_value)


def test_pad_or_crop(mri_image):
    image, header = mri_image
    assert image.shape == (39, 305, 305)

    fill_value = 100
    shape = (12, 400, 200)

    image_padded_cropped = tiger.resampling.pad_or_crop_image(
        image, target_shape=shape, fill=fill_value, align="min"
    )
    assert image_padded_cropped.shape == shape

    subimage = image_padded_cropped[:12, :305, :200]
    assert np.array_equal(subimage, image[:12, :, :200])

    subimage.fill(fill_value)
    assert np.all(image_padded_cropped == fill_value)


def test_pad_or_crop_errors(mri_image):
    image, header = mri_image

    # Different number of dimensions in target shape
    with pytest.raises(ValueError):
        tiger.resampling.pad_or_crop_image(image, target_shape=image.shape[:-1])

    # Align mode other than min/max/center
    with pytest.raises(ValueError):
        tiger.resampling.pad_or_crop_image(image, target_shape=(12, 12, 12), align="left")


def test_change_direction_simple():
    # Construct an image with non-identity direction cosine matrix
    image = np.arange(64**3).reshape((64, 64, 64))
    header = tiger.io.ImageMetadata(
        ndim=3,
        spacing=(0.3, 0.4, 0.5),
        origin=(134.23, 32.2, -425.45),
        direction=(0, 1, 0, -1, 0, 0, 0, 0, 1),
    )

    # Construct the header for the same image but with identity direction
    expected_header = tiger.io.ImageMetadata(
        ndim=3,
        spacing=(0.4, 0.3, 0.5),
        origin=(134.23, 13.3, -425.45),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    expected_header["original"] = header

    # Normalize image
    normalized_image, normalized_header = tiger.resampling.normalize_direction_simple(image, header)
    assert normalized_header == expected_header

    # Convert back from normalized direction to original direction
    restored_image, restored_header = tiger.resampling.restore_original_direction_simple(
        normalized_image, normalized_header
    )
    assert restored_image == pytest.approx(image)
    assert restored_header == header


def test_align_images():
    # Construct two images that are not aligned with each other
    image1 = np.arange(64**3).reshape((64, 64, 64))
    header1 = tiger.io.ImageMetadata(
        ndim=3,
        spacing=(0.3, 0.4, 0.5),
        origin=(134.23, 32.2, -425.45),
        direction=(0, 1, 0, -1, 0, 0, 0, 0, 1),
    )

    image2 = np.arange(82**3).reshape((82, 82, 82))
    header2 = tiger.io.ImageMetadata(
        ndim=3,
        spacing=(0.2, 0.2, 0.2),
        origin=(65.3, 16.4, -325.45),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )

    assert not header1.has_same_world_matrix(header2)

    # Align images and check if they have now the same dimensions and coordinate space
    new_image1, new_header1 = tiger.resampling.align_images(image1, header1, image2, header2)
    assert new_image1.shape == image2.shape
    assert new_header1.has_same_world_matrix(header2)


def test_weighted_average_resampling(ct_image, resources_path):
    image, header = ct_image
    slice_thickness = header["spacing"][2]

    war_image, war_header = tiger.io.read_image(resources_path / "ct_image_war.mha")
    target_slice_thickness = 3.0
    target_slice_spacing = 1.5

    # Create a resampler instance
    resampler = tiger.resampling.WeightedAverageResampler(
        target_slice_thickness, target_slice_spacing
    )

    # Resampler needs to know the slice thickness, will raise an exception
    with pytest.raises(ValueError):
        resampler.resample(image, header)

    # Specify slice thickness explicitly
    new_image, new_header = resampler.resample(image, header, slice_thickness=slice_thickness)
    assert new_image == pytest.approx(war_image)
    assert new_header == war_header

    # Specificically check spacing and slice thickness again
    assert new_header["spacing"][2] == target_slice_spacing
    assert new_header["slice_thickness"] == target_slice_thickness

    # Specify slice thickness as part of header
    header["slice_thickness"] = slice_thickness
    new_image, new_header = resampler.resample(image, header)
    assert new_image == pytest.approx(war_image)
    assert new_header == war_header
