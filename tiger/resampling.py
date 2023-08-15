import itertools
import warnings
from typing import Iterable, Optional, Tuple, Union, overload

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .io import Image, ImageMetadata, image_from_sitk, image_to_sitk


def sitk_interpolator(name: str):
    if name == "nearest":
        return sitk.sitkNearestNeighbor
    elif name == "linear":
        return sitk.sitkLinear
    elif name == "bspline":
        return sitk.sitkBSpline
    elif name == "gaussian":
        return sitk.sitkGaussian
    elif name == "lanczos":
        return sitk.sitkLanczosWindowedSinc
    else:
        raise ValueError(f'Unknown interpolation method "{name}"')


def resample_image(
    image: np.ndarray,
    spacing: Iterable[float],
    new_spacing: Iterable[float],
    *,
    order: int = 3,
    border_mode: str = "nearest",
    outside_val: float = 0,
    prefilter: bool = False,
) -> np.ndarray:
    """Resamples the image to a new voxel spacing using spline interpolation

    Parameters
    ----------
    image
        The original image

    spacing
        The original voxel spacing that the image has now

    new_spacing
        The desired voxel spacing to which the image will be resampled

    order
        Resampling is based on spline interpolation, 0 corresponds to nearest neighbor
        interpolation, 1 to linear interpolation and 3 to a cubic spline. Maximum order
        is 5.

    border_mode
        How values outside the image are handled during interpolation, can be nearest /
        reflect / constant / mirror / wrap.

    outside_val
        Value for voxels outside the image for border_mode = constant

    prefilter
        Whether the image is first filtered with a Gaussian filter - this can be beneficial
        when resampling images from high to much lower resolution. The interpolation order
        should then best be set to 0 = nearest neighbor.

    Returns
    -------
    Resampled image
    """
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))

    if prefilter:
        factor = 0.5 if type(prefilter) is bool else float(prefilter)
        sigmas = tuple((n / o) * factor for o, n in zip(spacing, new_spacing))
        image = ndimage.gaussian_filter(image, sigmas, mode=border_mode, cval=outside_val)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ndimage.zoom(
            image, resampling_factors, order=order, mode=border_mode, cval=outside_val
        )


def resample_image_itk(
    image: np.ndarray,
    spacing: Iterable[float],
    new_spacing: Iterable[float],
    *,
    outside_val: float = 0,
    interpolation: str = "linear",
    dtype: Optional[str] = None,
    rounding: bool = True,
    output_shape: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Resamples the image to a new voxel spacing using one of various interpolation methods (ITK implementation)"""
    interpolator = sitk_interpolator(interpolation)

    itkimg = image_to_sitk(image, ImageMetadata(ndim=image.ndim, spacing=spacing))
    if output_shape is None:
        output_shape = tuple(
            int(round(s * os / ns)) for s, os, ns in zip(image.shape, spacing, new_spacing)
        )

    itkimg = sitk.Resample(
        itkimg,
        output_shape,
        sitk.Transform(),
        interpolator,
        itkimg.GetOrigin(),
        new_spacing,
        itkimg.GetDirection(),
        outside_val,
        sitk.sitkFloat32,
    )
    resampled, _ = image_from_sitk(itkimg)

    npdtype = image.dtype if dtype is None else np.dtype(dtype)
    if rounding and npdtype.kind in ("i", "u"):
        resampled = np.round(resampled)
    return resampled.astype(npdtype)


def resample_mask(
    mask: np.ndarray, spacing: Iterable[float], new_spacing: Iterable[float]
) -> np.ndarray:
    """Resamples a label mask to a new voxel spacing using nearest neighbor interpolation

    Parameters
    ----------
    mask
        The original mask

    spacing
        The original voxel spacing of the provided mask

    new_spacing
        The desired new voxel spacing

    Returns
    -------
    Mask resampled to new_spacing using nearest neighbor interpolation
    """
    return resample_image(mask, spacing, new_spacing, order=0, border_mode="constant")


def resample_mask_itk(
    mask: np.ndarray,
    spacing: Iterable[float],
    new_spacing: Iterable[float],
    *,
    output_shape: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Resamples a label mask to a new voxel spacing using nearest neighbor interpolation (ITK implementation)"""
    return resample_image_itk(
        mask, spacing, new_spacing, interpolation="nearest", output_shape=output_shape
    )


def resample_mask_dt(
    mask: np.ndarray, spacing: Iterable[float], new_spacing: Iterable[float], **kwargs
) -> np.ndarray:
    """Resamples a label mask using a distance transform"""
    resampled = None  # can't initialize the resampled mask yet since we don't know the shape
    threshold = np.linalg.norm(new_spacing)

    labels = set(np.unique(mask))
    for label in labels - {0}:
        binary_mask = mask == label

        # PyCharm does not correctly understand that distance_transform_edt accepts an iterable
        # as parameter for "sampling" (does not correctly interpret the docstring) -> ignore
        # noinspection PyTypeChecker
        dt = ndimage.distance_transform_edt(~binary_mask, sampling=spacing)

        dt_resampled = resample_image(dt, spacing, new_spacing, **kwargs)
        binary_mask_resampled = abs(dt_resampled) < threshold

        if resampled is None:
            resampled = np.zeros_like(binary_mask_resampled, dtype=mask.dtype)
        resampled[binary_mask_resampled] = label

    return resampled


def pad_or_crop_image(
    image: np.ndarray,
    target_shape: Iterable[int],
    fill: float = 0,
    align: Union[str, Iterable[str]] = "center",
) -> np.ndarray:
    """Ensures that an image has a specific shape by padding or cropping if necessary

    Parameters
    ----------
    image
        An image or a mask with an arbitray number of dimensions

    target_shape
        The desired shape of the image after padding/cropping

    fill
        Value that will be used for padding

    align
        Where the image will be padded or cropped - the default value "center" means that the image will be
        equally padded or cropped on either side (an input image that is 3x3 will be padded to a 5x5 target
        shape by adding the fill value once in front and once at the end of each axis). Alternative values
        are "min" (pad or crop at the beginning of the axis) or "max" (pad or crop at the end of the axis).
        If a single mode is specified, that mode is used for all axes, but it is possible to specify the
        mode per axis and to use different modes for different axes.

    Returns
    -------
    The input image padded or cropped to target_shape
    """
    image = np.atleast_1d(image)
    target_shape = tuple(target_shape)

    if len(target_shape) != image.ndim:
        raise ValueError(
            "Dimension mismatch between input and target shape, "
            "expected {} dimensions, got {}".format(image.ndim, target_shape)
        )

    aligns = [align] * image.ndim if isinstance(align, str) else list(align)
    for align in aligns:
        if align not in ("min", "max", "center"):
            raise ValueError(
                f'Unsupported align mode "{align}", expected one of "min", "max", "center"'
            )

    new_image = np.ones(shape=target_shape, dtype=image.dtype) * fill

    pads = [0] * image.ndim
    crops = [0] * image.ndim
    for axis in range(image.ndim):
        if image.shape[axis] < target_shape[axis]:
            if aligns[axis] == "min":
                pads[axis] = 0
            elif aligns[axis] == "max":
                pads[axis] = target_shape[axis] - image.shape[axis]
            else:
                pads[axis] = (target_shape[axis] - image.shape[axis]) // 2
        elif image.shape[axis] > target_shape[axis]:
            if aligns[axis] == "min":
                crops[axis] = 0
            elif aligns[axis] == "max":
                crops[axis] = image.shape[axis] - target_shape[axis]
            else:
                crops[axis] = (image.shape[axis] - target_shape[axis]) // 2

    cropped = image[
        tuple(slice(crop, crop + axis_length) for crop, axis_length in zip(crops, target_shape))
    ]
    new_image[
        tuple(
            slice(padding, padding + cropped_axis_length)
            for padding, cropped_axis_length in zip(pads, cropped.shape)
        )
    ] = cropped

    return new_image


def reorient_image(
    image: np.ndarray,
    header: ImageMetadata,
    new_direction: Iterable[float] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    *,
    new_spacing: Optional[Tuple[float, float, float]] = None,
    new_shape: Optional[Tuple[int, int, int]] = None,
    outside_val: float = 0,
    interpolation: Optional[str] = None,
    eps: float = 0.001,
) -> Image:
    """Changes the orientation (direction cosine matrix) of an image

    This corresponds to changing the meaning of a "slice" in a 3D tomographic image, e.g.,
    from coronal to axial slices. However, the function supports arbitray orientations like
    there are sometimes seen in MR images. The function automatically determines whether
    interpolation is required.

    Parameters
    ----------
    image
        The input image that will be resampled. Either a numpy array accompanied by an
        ImageMetadata header, or a SimpleITK image.

    header
        The header of the image that contains the original direction cosine matrix as well
        as information about the voxel spacing and the origin of the coordinate system.

    new_direction
        Row major (or 3x3) cosine direction matrix, has to be a valid 3D rotation matrix

    new_spacing
        Whether to keep the original spacing or compute the new spacing. None means that
        the new spacing is automatically calculated. Alternatively, the new spacing can
        be supplied.

    new_shape
        Shape of the resampled image, if left empty (None) the shape will be computed
        from the input shape.

    outside_val
        Value used for voxels outside the image during resampling

    interpolation
        Interpolation mode, can be one of "nearest" for nearest neighbor interpolation,
        "linear", "bspline", "gaussian" or "lanczos". If none of these is specified,
        nearest neighbor interpolation is used for orientation changes that only involve
        inverting and transposing axis, and linear interpolation otherwise.

    eps
        Epsilon value used for comparisons, can be used to influence the tolerance toward
        what kind of matrices are considered orthogonal. Direction cosine matrices read
        from image headers are not always perfectly orthogonal due to floating point issues,
        so sometimes it can make sense to use a large epsilon value.

    Returns
    -------
    Resampled image, as numpy array + ImageMetadata if that was also the input, or SimpleITK Image otherwise
    """
    # If a header was supplied, assume that the image is a numpy array (otherwise a simpleitk image)
    if header is not None:
        image = image_to_sitk(image, header)

    if image.GetDimension() != 3:
        raise ValueError("Image is not 3D, only 3D images are supported at the moment")

    # Verify that the transformation matrix is a rotation matrix
    new_direction = np.asarray(new_direction).reshape(3, 3)
    if not (
        np.allclose(new_direction.dot(new_direction.T), np.eye(3), rtol=0, atol=eps)
        and np.allclose(new_direction.T.dot(new_direction), np.eye(3), rtol=0, atol=eps)
        and abs(np.linalg.det(new_direction) - 1) < eps
    ):
        raise ValueError("Desired direction is not a valid rotation matrix")

    # Assemble transformation
    transform = sitk.VersorTransform()
    transform.SetMatrix([float(s) for s in new_direction.flatten()])
    inverse_transform = transform.GetInverse()

    # Compute image bounding box in world coordinates and transform into new coordinate system
    extreme_points = [
        inverse_transform.TransformPoint(image.TransformIndexToPhysicalPoint(corner))
        for corner in itertools.product(*([0, s - 1] for s in image.GetSize()))
    ]

    # Compute new origin
    new_origin = transform.TransformPoint(np.min(extreme_points, axis=0))

    # Compute new spacing if needed
    if new_spacing is None:
        # To determine the new spacing, transform the two voxels (0, 0, 0) and (1, 1, 1)
        # to the new coordinate system and subtract them
        points = [
            np.asarray(inverse_transform.TransformPoint(image.TransformIndexToPhysicalPoint(voxel)))
            for voxel in [(0, 0, 0), (1, 1, 1)]
        ]
        new_spacing = abs(points[1] - points[0])
    else:
        new_spacing = np.asarray(new_spacing, dtype=float)

    # Compute new shape based on image size in mm and voxel spacing
    if new_shape is None:
        # The size of the image are the corner voxels plus one time the spacing - the coordinates of
        # the corners are the centers of the voxels, so they have half of the spacing to the edge of
        # the image around them
        image_size = (
            abs(np.max(extreme_points, axis=0) - np.min(extreme_points, axis=0)) + new_spacing
        )
        new_shape = np.rint(np.abs(image_size / new_spacing))

    # Choose a suitable interpolator
    if interpolation is None:
        interpolator = sitk.sitkNearestNeighbor
        vv = list(image.GetDirection()) + new_direction.flatten().tolist()
        if any(abs(v) > eps and abs(abs(v) - 1) > eps for v in vv):
            interpolator = sitk.sitkLinear
    else:
        interpolator = sitk_interpolator(interpolation)

    # Resample image
    resampled = sitk.Resample(
        image,
        [int(s) for s in new_shape],
        sitk.Transform(),
        interpolator,
        [float(o) for o in new_origin],
        [float(s) for s in new_spacing],
        [float(s) for s in new_direction.flatten()],
        outside_val,
        image.GetPixelID(),
    )

    # If the input was a numpy array, return also a numpy array (simpleitk image otherwise)
    if header is None:
        return resampled
    else:
        return image_from_sitk(resampled)


@overload
def _swap_flip_dimensions(cosine_matrix: np.ndarray, image: np.ndarray) -> np.ndarray:
    ...


@overload
def _swap_flip_dimensions(
    cosine_matrix: np.ndarray, image: np.ndarray, header: ImageMetadata = ...
) -> Image:
    ...


def _swap_flip_dimensions(cosine_matrix, image, header=None):
    # Compute swaps and flips
    swap = np.argmax(abs(cosine_matrix), axis=0)
    flip = np.sum(cosine_matrix, axis=0)

    # Apply transformation to image volume
    image = np.transpose(image, tuple(swap))
    image = image[tuple(slice(None, None, int(f)) for f in flip)]

    if header is None:
        return image

    # Apply transformation to header (assuming that we are transforming to RAI orientation)
    header["origin"] = np.min(
        [
            header.indices_to_physical_coordinates((0, 0, 0)),
            header.indices_to_physical_coordinates([s - 1 for s in image.shape]),
        ],
        axis=0,
    )
    header["spacing"] = tuple(header["spacing"][s] for s in swap)
    header["direction"] = np.identity(3)

    return image, header


def normalize_direction_simple(image: np.ndarray, header: ImageMetadata) -> Image:
    """
    Similar to the change_direction function, this function changes the direction / orientation of the image.
    However, this function is a simpler implementation without interpolation. Only the transformation of sagittal
    or coronal images into axial images is supported. Slices are only reorganized, without any interpolation.
    The original orientation is stored in the header and can be restored with the restore_original_direction_simple
    function. Use the is_common_direction function to check if the image is suitable for this function or whether
    the less efficient change_direction function has to be used.
    """
    # Preserve original header so that we can easily transform back
    original_header = header
    header = header.copy()
    header["original"] = original_header

    # Compute inverse of cosine (round first because we assume 0/1 values only)
    # to determine how the image has to be transposed and flipped for cosine = identity
    cosine = np.asarray(header["direction"]).reshape(3, 3)
    cosine_inv = np.round(cosine).T

    return _swap_flip_dimensions(cosine_inv, image, header)


def restore_original_direction_simple(mask: np.ndarray, header: ImageMetadata) -> Image:
    """Restores the original orientation (sagittal or coronal) of an image that has previously been normalized"""
    # Use original orientation for transformation because we assume the image to be in
    # normalized orientation, i.e., identity cosine)
    cosine = np.asarray(header["original"]["direction"]).reshape(3, 3)
    cosine_rnd = np.round(cosine)

    # Apply transformations to both the image and the mask
    return _swap_flip_dimensions(cosine_rnd, mask), header["original"]


def align_images(
    image: np.ndarray,
    header: ImageMetadata,
    reference_shape: Union[Iterable[int], np.ndarray],
    reference_header: ImageMetadata,
    interpolation: str = "linear",
    outside_val: float = 0,
) -> Image:
    """Resample the image to match the coordinate space of the reference image"""
    if header.has_same_world_matrix(reference_header):
        return image, header

    try:
        reference_shape = reference_shape.shape
    except AttributeError:
        pass

    itk_image = image_to_sitk(image, header)
    itk_resampled = sitk.Resample(
        itk_image,
        [int(s) for s in reference_shape],
        sitk.Transform(),
        sitk_interpolator(interpolation),
        [float(o) for o in reference_header.origin],
        [float(s) for s in reference_header.spacing],
        [float(d) for d in reference_header.direction.flatten()],
        outside_val,
        itk_image.GetPixelID(),
    )
    return image_from_sitk(itk_resampled)


def align_mask_with_image(
    mask: np.ndarray,
    header: ImageMetadata,
    reference_shape: Union[Iterable[int], np.ndarray],
    reference_header: ImageMetadata,
) -> Image:
    """Resamples the mask to match the coordinate space of the reference image"""
    return align_images(
        mask, header, reference_shape, reference_header, interpolation="nearest", outside_val=0
    )


class WeightedAverageResampler:
    """Resamples images using weighted averaging to simulate partial volume effects"""

    def __init__(self, target_slice_thickness: float, target_slice_spacing: float):
        self.target_slice_thickness = target_slice_thickness
        self.target_slice_spacing = target_slice_spacing

    def resample(
        self, image: np.ndarray, header: ImageMetadata, slice_thickness: Optional[float] = None
    ) -> Image:
        """
        Resamples an image to a new slice spacing and slice thickness.
        The slice thickness of the input image needs to be specified, either in the parameter
        slice_thickness or under the key 'slice_thickness' in the header dictionary.
        Resampling from thick to thin slices should be avoided.

        Parameters
        ----------
        image
            The pixel data of the image

        header
            Needs to contain spacing and origin, and optionally slice_thickness

        slice_thickness
            Slice thickness (not spacing!) of the image

        Returns
        -------
        pixel_data
            The resampled image
        header
            The corresponding metadata

        Raises
        ------
        ValueError
            If the slice thickness of the input image cannot be determined, i.e., it wasn't explicitly
            provided and could also not be found in the header.
        """
        if slice_thickness is None:
            try:
                slice_thickness = header["slice_thickness"]
            except KeyError as e:
                raise ValueError(
                    "Value for slice thickness expected, but neither explicitly supplied nor attached to header"
                ) from e

        slice_spacing = abs(float(header["spacing"][2]))
        slice_thickness = abs(float(slice_thickness))

        # Compute number of slices in resampled image
        scan_thickness = slice_thickness + (slice_spacing * (image.shape[2] - 1))
        target_num_slices = int(
            np.floor((scan_thickness - self.target_slice_thickness) / self.target_slice_spacing + 1)
        )

        # Compute offset of the origin in the new image
        origin_offset_z = -(0.5 * slice_thickness) + (0.5 * self.target_slice_thickness)

        # Create a new (empty) image volume
        target_shape = (image.shape[0], image.shape[1], target_num_slices)
        resampled_image = np.empty(shape=target_shape, dtype=image.dtype)

        resampled_header = header.copy()
        resampled_header["spacing"] = (
            header["spacing"][0],
            header["spacing"][1],
            self.target_slice_spacing,
        )
        resampled_header["origin"] = (
            header["origin"][0],
            header["origin"][1],
            header["origin"][2] + origin_offset_z,
        )
        resampled_header["slice_thickness"] = self.target_slice_thickness

        # Fill new image with values
        for z in range(resampled_image.shape[2]):
            sum_weights = 0.0
            sum_values = np.zeros((resampled_image.shape[0], resampled_image.shape[1]), dtype=float)

            slice_begin = z * self.target_slice_spacing
            slice_end = slice_begin + self.target_slice_thickness

            # Find first slice in the old image that overlaps with the new slice
            old_slice = 0
            old_slice_begin = 0.0
            old_slice_end = slice_thickness

            while old_slice_end < slice_begin:
                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing

            # Find all slices in the old image that overlap with the new slice
            while old_slice < image.shape[2] and old_slice_begin < slice_end:
                if old_slice_end <= slice_end:
                    weight = (old_slice_end - max(slice_begin, old_slice_begin)) / slice_thickness
                    sum_weights += weight
                    sum_values += weight * image[:, :, old_slice]
                elif old_slice_begin >= slice_begin:
                    weight = (slice_end - old_slice_begin) / slice_thickness
                    sum_weights += weight
                    sum_values += weight * image[:, :, old_slice]

                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing

            resampled_image[:, :, z] = np.round(sum_values / sum_weights)

        return resampled_image, resampled_header
