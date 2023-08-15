from typing import Iterable, Optional, Union

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates, rotate

from .random import random_decision
from .utils import repeat_if_needed


def random_elastic_transform(
    image,
    mask=None,
    alpha: float = 2.5,
    sigma: float = 10,
    interpolation_order: int = 1,
):
    """
    Applies random elastic transformations to the image (which can be 2D or 3D)

    Parameters
    ----------
    image : np.ndarray or iterable of np.ndarray
        2D or 3D image. If multiple images of the same shape are supplied, the same transformations are applied to all of them.

    mask : np.ndarray or iterable of np.ndarray or None
        2D or 3D mask of the same shape as the image. The same transformations will be applied to both image and mask.

    alpha
        Maximum magnitude of displacement vectors

    sigma
        Smoothness factor

    interpolation_order
        Spline order used for interpolation, should be 0-5. For the mask, this is always 0 (=nearest neighbor interpolation)

    Returns
    -------
    Either only the deformed image or the deformed image and the deformed mask, if a mask was supplied. If multiple images and/or
    masks were supplied, a list of images and/or masks is returned instead of a single image/mask.
    """
    multiple_images = not isinstance(image, np.ndarray)
    multiple_masks = not (mask is None or isinstance(mask, np.ndarray))

    images = list(image) if multiple_images else [image]
    if mask is None:
        masks = []
    else:
        masks = list(mask) if multiple_masks else [mask]

    shape = images[0].shape
    if len(images) > 1:
        for image in images[1:]:
            if image.shape != shape:
                raise ValueError(
                    f"All images must have the same shape, got {image.shape} and {shape}"
                )
    for mask in masks:
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} != image shape {shape}")

    original_indices = [np.arange(s) for s in shape]
    original_grid = np.meshgrid(*original_indices, indexing="ij")

    new_indices = []
    for indices in original_grid:
        displacements = (
            gaussian_filter(np.random.rand(*shape) * 2 - 1, sigma, mode="constant", cval=0) * alpha
        )
        new_indices.append(np.reshape(indices + displacements, -1))

    deformed_images = [
        map_coordinates(image, new_indices, order=interpolation_order, mode="reflect")
        .reshape(shape)
        .astype(image.dtype)
        for image in images
    ]
    retval_images = deformed_images if multiple_images else deformed_images[0]

    if mask is None:
        return retval_images
    else:
        deformed_masks = [
            map_coordinates(mask, new_indices, order=0, mode="constant", cval=0)
            .reshape(shape)
            .astype(mask.dtype)
            for mask in masks
        ]
        retval_masks = deformed_masks if multiple_masks else deformed_masks[0]
        return retval_images, retval_masks


def random_rotation(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: Union[float, Iterable[float]] = 10,
    axes: Optional[Iterable[int]] = None,
):
    """
    Applies random rotations to the image (which can be 2D or 3D)

    Parameters
    ----------
    image
        2D or 3D image

    mask
        2D or 3D mask of the same shape as the image. The same transformations will be applied to both image and mask.

    sigma
        Rotation angles are drawn from a normal distribution with this standard deviation. Can be either a single scalar
        that is used for all axes, or an iterable with one angle per axis.

    axes
        List of axes about which to rotate. Only relevant for 3D inputs. 0 = Sagittal, 1 = Coronal, 2 = Axial. If no list
        is supplied (None), the image is rotated about all three axes.

    Returns
    -------
    Either only the deformed image or the deformed image and the deformed mask, if a mask was supplied.
    """
    image = np.asarray(image)
    assert mask is None or (isinstance(mask, np.ndarray) and mask.shape == image.shape)
    assert image.ndim in (2, 3)

    # 2D image
    if image.ndim == 2:
        rotation_angle = np.random.normal(scale=sigma)
        deformed_image = rotate(image, rotation_angle, reshape=False, mode="reflect")
        if mask is None:
            return deformed_image
        else:
            deformed_mask = rotate(
                mask, rotation_angle, reshape=False, order=0, mode="constant", cval=0
            )
            return deformed_image, deformed_mask

    # 3D volume
    if axes is None:
        axes = (0, 1, 2)  # rotated about all three axes
    else:
        axes = tuple(axes)

    sigmas = repeat_if_needed(sigma, len(axes))

    deformed_image = image
    deformed_mask = mask
    for axis, sigma in zip(axes, sigmas):
        rotation_axes = [i for i in range(3) if i != axis]
        rotation_angle = np.random.normal(scale=sigma)

        deformed_image = rotate(
            deformed_image,
            angle=rotation_angle,
            axes=rotation_axes,
            reshape=False,
            mode="reflect",
        )
        if mask is not None:
            deformed_mask = rotate(
                deformed_mask,
                angle=rotation_angle,
                axes=rotation_axes,
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )

    if mask is None:
        return deformed_image
    else:
        return deformed_image, deformed_mask


def random_flip(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    probability: Union[float, Iterable[float]] = 0.5,
    axes: Optional[Iterable[int]] = None,
):
    """
    Randomly flips an image

    Parameters
    ----------
    image
        Image of any dimensionality

    mask
        Mask of the same shape as the image. The same transformations will be applied to both image and mask.

    probability
        Probability per axis that the image is flipped/mirrored. Can be a single scalar, then all axes have the same probability.

    axes
        List of axes that are randomly inverted. If no value is supplied (None), all axes are potentially inverted.

    Returns
    -------
    Either only the deformed image or the deformed image and the deformed mask, if a mask was supplied.
    """
    image = np.asarray(image)
    assert mask is None or (isinstance(mask, np.ndarray) and mask.shape == image.shape)

    if axes is None:
        axes = tuple(range(image.ndim))
    else:
        axes = tuple(axes)

    probabilities = repeat_if_needed(probability, image.ndim)
    indices = tuple(
        slice(None, None, -1 if i in axes and random_decision(p) else None)
        for i, p in zip(range(image.ndim), probabilities)
    )

    if mask is None:
        return np.copy(image[indices])
    else:
        return np.copy(image[indices]), np.copy(mask[indices])


def random_gaussian_noise(
    image: np.ndarray, sigma: float = 100, random_sigma: bool = True
) -> np.ndarray:
    """
    Adds random white noise to the image

    Parameters
    ----------
    image
        Image of any dimensionality

    sigma
        Standard deviation of the Gaussian noise that is added to the image

    random_sigma
        If this parameter is True, the sigma is randomly (uniformly) choosen in the range [0,sigma].

    Returns
    -------
    image
        Image with added noise
    """
    image = np.asarray(image)

    if random_sigma:
        sigma = np.random.rand() * sigma

    noise = np.random.normal(size=image.shape, scale=sigma)
    image_with_noise = image + noise

    if np.issubdtype(image.dtype, np.integer):
        return np.round(image_with_noise).astype(image.dtype)
    else:
        return image_with_noise.astype(image.dtype)


def random_gaussian_smoothing(
    image: np.ndarray, sigma: float = 3, min_random_sigma: float = 0.1
) -> np.ndarray:
    """
    Apply Gaussian smoothing with random sigma

    Parameters
    ----------
    image
        Image of any dimensionality

    sigma
        Standard deviation of the distribution from which the sigma for the
        Gaussian smoothing that is applied to the image is sampled

    min_random_sigma
        Minimum value for which smoothing is still applied. Internally, the function
        chooses a random sigma value for the Gaussian smoothing, but if this value is
        very small, the effect on the image would be minimal.

    Returns
    -------
    image
        Smoothed image
    """
    image = np.asarray(image)

    random_sigma = abs(np.random.normal(scale=sigma)) / 2  # half normal distribution
    if random_sigma < min_random_sigma:
        return image
    else:
        image_floating = (
            image if np.issubdtype(image.dtype, np.floating) else image.astype("float32")
        )
        image_smoothed = gaussian_filter(image_floating, sigma=random_sigma)

        if np.issubdtype(image.dtype, np.integer):
            return np.round(image_smoothed).astype(image.dtype)
        else:
            return image_smoothed.astype(image.dtype)
