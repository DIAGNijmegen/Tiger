import itertools
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, rotate

from .utils import repeat_if_needed

try:
    from torch.utils.data import IterableDataset
except ImportError:
    from collections.abc import Iterable as IterableDataset


def compute_crop(coordinate: int, width: int, axis_length: int) -> Tuple[int, int, int, int]:
    """Computes beginning and end coordinates and padding of a patch in one dimension based on patch size and axis length

    If the patch would partially be outside the image, some padding would be needed to achieve the target size (width).
    This function computes the start and end coordinates (indices) and the needed padding.

    Parameters
    ----------
    coordinate
        Center of the patch along one of the axes (for an image with shape [10, 20, 30],
        the coordinate could be 22 for computing the crop parameters for the z axis for
        a patch centered at (?, ?, 22).
    width
        Size of the patch along the axes (for a 2x4x6 patch, this would be 6 for the z axis)
    axis_length
        The size of the image (30 for the z axis in the example above)

    Returns
    -------
    from
        Start coordinate of the patch
    to
        End coordinate of the patch
    padding_before
        Padding that should be added before the extracted patch
    padding_after
        Padding that should be added after the extracted patch
    """
    coordinate_from = coordinate - (width // 2)
    coordinate_to = coordinate + int(np.ceil(width / 2.0))

    padding_before = 0 if coordinate_from >= 0 else abs(coordinate_from)
    padding_after = 0 if coordinate_to < axis_length else coordinate_to - axis_length

    if padding_before > 0:
        coordinate_from = 0

    return coordinate_from, coordinate_to, padding_before, padding_after


def compute_valid_patch_pairs(
    mask_shape: Iterable[int], patch_shape: Iterable[int], patch_center: Iterable[int]
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """Computes the offset of two patches of equal size corresponding to the valid area of the supplied patch"""
    crops = [compute_crop(c, s, a) for c, s, a in zip(patch_center, patch_shape, mask_shape)]
    mask_offset = tuple(slice(c[0], c[1]) for c in crops)
    patch_offset = tuple(
        slice(c[2] if c[2] > 0 else None, -c[3] if c[3] > 0 else None) for c in crops
    )
    return mask_offset, patch_offset


def extract_valid_patch_pairs(
    mask: np.ndarray, patch: np.ndarray, patch_center: Iterable[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts two patches of equal size corresponding to the valid area of the supplied mask patch"""
    mask_offset, patch_offset = compute_valid_patch_pairs(mask.shape, patch.shape, patch_center)
    return mask[mask_offset], patch[patch_offset]


class PatchExtractor3D:
    """
    Extracts 2D, 2.5D and 3D patches from a 3D volume

    Parameters
    ----------
    image
        3D image volume

    voxel_spacing
        Spacing between voxels in the image

    pad_value
        Value used to pad extracted patches if they extent beyond the image boundaries

    spline_order
        Order of the spline used for interpolation (if necessary)

    dtype
        Datatype of the extracted patches, 'float32' is usually a good value.
        Defaults to the dtype of the image.
    """

    sagittal = 0  #: Value for axis parameter to extract sagittal patches/slices
    coronal = 1  #: Value for axis parameter to extract coronal patches/slices
    axial = 2  #: Value for axis parameter to extract axial patches/slices

    def __init__(
        self,
        image: np.ndarray,
        voxel_spacing: Optional[Tuple[float, float, float]] = None,
        pad_value: float = 0,
        spline_order: int = 1,
        dtype: Optional[str] = None,
    ):
        self.image = np.asarray(image)
        if self.image.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume, got array with {self.image.ndim} dimensions instead"
            )

        self.voxel_spacing = voxel_spacing
        self.pad_value = pad_value
        self.spline_order = int(spline_order)

        if dtype is None:
            self.dtype = self.image.dtype
        else:
            self.dtype = np.dtype(dtype)

        # Store orthogonal views
        self.views = (
            np.transpose(self.image, axes=(1, 2, 0)),  # sagittal (y, z, x)
            np.transpose(self.image, axes=(0, 2, 1)),  # coronal  (x, z, y)
            self.image,  # axial    (x, y, z)
        )

    def _interpolate(
        self,
        image: np.ndarray,
        image_spacing: Iterable[float],
        patch_center: Iterable[float],
        patch_shape: Iterable[int],
        patch_extent: Iterable[float],
    ) -> np.ndarray:
        patch_spacing = [e / s for e, s in zip(patch_extent, patch_shape)]

        sample_radius = [
            (ps / cs) * (s / 2.0) for ps, cs, s in zip(patch_spacing, image_spacing, patch_shape)
        ]
        sample_points = [
            np.linspace(pc - sr, pc + sr, s)
            for pc, sr, s in zip(patch_center, sample_radius, patch_shape)
        ]

        chunk_from = [max(0, int(np.floor(sp[0])) - 1) for sp in sample_points]
        chunk_to = [int(np.ceil(sp[-1])) + 1 for sp in sample_points]
        chunk = image[tuple(slice(cf, ct) for cf, ct in zip(chunk_from, chunk_to))]

        sample_grid = np.meshgrid(*[sp - cf for sp, cf in zip(sample_points, chunk_from)])
        return map_coordinates(
            chunk,
            sample_grid,
            order=self.spline_order,
            output=self.dtype,
            cval=self.pad_value,
        )

    def extract_slice(self, i: int, axis: int = 2) -> np.ndarray:
        """Returns an entire slice (sagittal, coronal or axial view depending on the specified axis)"""
        s = self.views[axis][:, :, i]
        if axis != 2:
            s = np.fliplr(s)
        return s

    def extract_rect(
        self,
        center_voxel: Tuple[int, int, int],
        shape: Tuple[int, int],
        extent: Tuple[float, float] = None,
        axis: int = 2,
        rotation_angle: Optional[float] = None,
    ) -> np.ndarray:
        """Extracts a 2D rectangular patch

        Parameters
        ----------
        center_voxel
            Coordinates of the patch center (indices)
        shape
            Patch shape in pixels, e.g. (20, 40) for a 20x40 patch
        extent
            The physical size of the patch in mm (optional)
        axis
            Sagittal, coronal or axial view (0-2)
        rotation_angle
            If an angle is specified, a larger patch is extracted, rotated and cropped to the desired size

        Returns
        -------
        patch
            The 2D patch

        Raises
        ------
        ValueError
            If "extent" is specified but the voxel spacing was not configured when creating the patch extractor instance.
        """
        if rotation_angle is not None:
            # Extract a larger patch, rotate and crop
            radius = int(np.ceil(np.linalg.norm(shape, ord=2)))

            if extent:
                radius += (
                    radius % 2
                )  # not sure why this is necessary, but without it the patch is slightly shifted
                patch_spacing = [e / s for e, s in zip(extent, shape)]
                extent = tuple(radius * ps for ps in patch_spacing)  # type: ignore

            patch_with_padding = self.extract_rect(center_voxel, (radius, radius), extent, axis)
            rotated_patch = rotate(
                patch_with_padding,
                angle=rotation_angle,
                reshape=False,
                order=self.spline_order,
                output=self.dtype,
                cval=self.pad_value,
            )

            patch_from = [(rps - s) // 2 for rps, s in zip(rotated_patch.shape, shape)]
            patch_to = [pf + s for pf, s in zip(patch_from, shape)]
            return rotated_patch[tuple(slice(pf, pt) for pf, pt in zip(patch_from, patch_to))]

        # Extract 2D slab orthogonal to the specified axis
        slab = self.views[axis][:, :, center_voxel[axis]]

        if axis == 0:
            i, j = 1, 2
        elif axis == 1:
            i, j = 0, 2
        else:
            i, j = 0, 1

        if extent:
            # Extract patch with interpolation
            if self.voxel_spacing is None:
                raise ValueError(
                    "Cannot perform interpolation if the voxel spacing of the image is unknown"
                )

            slab_spacing = (self.voxel_spacing[i], self.voxel_spacing[j])
            patch_center = (center_voxel[i], center_voxel[j])
            patch = self._interpolate(slab, slab_spacing, patch_center, shape, extent).T
        else:
            # Extract patch without interpolation
            i_from, i_to, i_padding_before, i_padding_after = compute_crop(
                center_voxel[i], shape[0], slab.shape[0]
            )
            j_from, j_to, j_padding_before, j_padding_after = compute_crop(
                center_voxel[j], shape[1], slab.shape[1]
            )

            # Generate array with the proper size by padding with pad_value
            chunk = slab[i_from:i_to, j_from:j_to]
            paddings = (
                (i_padding_before, i_padding_after),
                (j_padding_before, j_padding_after),
            )
            patch = np.pad(chunk, paddings, mode="constant", constant_values=self.pad_value).astype(
                self.dtype
            )

        # Make sure the patch orientation is correct
        if axis != 2:
            patch = np.fliplr(patch)  # Flip axis=1

        return patch

    def extract_ortho_rects(
        self,
        center_voxel: Tuple[int, int, int],
        shape: Tuple[int, int],
        extent: Optional[
            Union[
                Tuple[float, float],
                Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
            ]
        ] = None,
        stack_axis: int = 0,
        rotation_angle: Optional[Union[float, Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """Extracts a set of three orthogonal 2D patches (sagittal, coronal, axial)

        Parameters
        ----------
        center_voxel
            Coordinates of the patch center (indices)
        shape
            Patch shape in pixels, e.g. (20, 40) for three 20x40 patches
        extent
            The physical size of the patch in mm, can either be one size or three individual sizes (optional)
        stack_axis
            The function extracts three patches and stacks them along this axis
        rotation_angle
            If an angle is specified, a larger patch is extracted, rotated and cropped to the desired size. Can
            be either a single value which is then used for all three patches, or three different rotation angles
            for the sagittal, coronal and axial patches.

        Returns
        -------
        patches
            The stack of 2D patches

        Raises
        ------
        ValueError
            If "extent" is specified but the voxel spacing was not configured when creating the patch extractor instance.
        """
        if extent is None:
            extents = (None, None, None)
        else:
            n_extent = len(extent)
            if n_extent == 2:
                extents = itertools.repeat(
                    extent, 3
                )  # 2 values given, so use those for all orientations
            elif n_extent != 3:
                raise ValueError(
                    f'Expected two values for "extent", or three times two values, but got "{extent}"'
                )
            else:
                extents = extent

        if rotation_angle is not None:
            angles = repeat_if_needed(rotation_angle, 3)
        else:
            angles = (None, None, None)

        return np.stack(
            [
                self.extract_rect(
                    center_voxel,
                    shape,
                    extent=extents[i],
                    axis=i,
                    rotation_angle=angles[i],
                )
                for i in range(3)
            ],
            axis=stack_axis,
        )

    def extract_cuboid(
        self,
        center_voxel: Tuple[int, int, int],
        shape: Tuple[int, int, int],
        extent: Optional[Tuple[float, float, float]] = None,
        rotation_angles: Optional[Union[float, Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """Extracts a 3D patch

        Parameters
        ----------
        center_voxel
            The coordinates of the patch center (indices)
        shape
            The size of the patch, e.g., (30, 20, 10) for a 30x20x10 patch
        extent
            The physical size of the patch in mm  (optional)
        rotation_angles
            Either a single angle or one rotation angle per x, y, and z axis. The rotations are combined into
            a single 3D rotation matrix, which is then applied to a larger patch to avoid border artifacts.
            The rotated patch is then cropped to the desired patch shape.

        Returns
        -------
        patch
            The 3D patch

        Raises
        ------
        ValueError
            If "extent" is specified but the voxel spacing was not configured when creating the patch extractor instance.
        """
        if rotation_angles is not None:
            # Extract patch with extra padding
            radius = int(np.ceil(np.linalg.norm(shape, ord=2)))

            if extent is not None:
                patch_spacing = [e / s for e, s in zip(extent, shape)]
                extent = tuple(radius * ps for ps in patch_spacing)

            patch_with_padding = self.extract_cuboid(center_voxel, (radius, radius, radius), extent)

            # Assemble 3D rotation matrix from three angles (about-x-axis, about-y-axis, about-z-axis)
            rads = [np.deg2rad(angle) for angle in repeat_if_needed(rotation_angles, 3)]

            c, s = np.cos(rads[0]), np.sin(rads[0])
            rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

            c, s = np.cos(rads[1]), np.sin(rads[1])
            rot_y = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

            c, s = np.cos(rads[2]), np.sin(rads[2])
            rot_z = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

            rot_matrix = rot_z @ rot_y @ rot_x

            # Compute center offset between padded patch and rotated output patch
            in_center = (np.array(patch_with_padding.shape) - 1) / 2
            out_center = rot_matrix @ ((np.array(shape) - 1) / 2)
            offset = tuple(in_center - out_center)

            # Apply rotation matrix
            return affine_transform(
                patch_with_padding,
                rot_matrix,
                offset,
                shape,
                order=self.spline_order,
                output=self.dtype,
                cval=self.pad_value,
            )

        if extent:
            # Extract patch with interpolation
            if self.voxel_spacing is None:
                raise ValueError(
                    "Cannot perform interpolation if the voxel spacing of the image is unknown"
                )

            patch = self._interpolate(self.image, self.voxel_spacing, center_voxel, shape, extent)
            patch = np.flipud(np.rot90(patch))
        else:
            # Extract patch without interpolation
            crops = [
                compute_crop(cv, s, al) for cv, s, al in zip(center_voxel, shape, self.image.shape)
            ]
            chunk = self.image[
                crops[0][0] : crops[0][1],
                crops[1][0] : crops[1][1],
                crops[2][0] : crops[2][1],
            ]
            paddings = tuple(c[2:4] for c in crops)
            patch = np.pad(chunk, paddings, mode="constant", constant_values=self.pad_value).astype(
                self.dtype
            )

        return patch


class PatchExtractor2D:
    """
    Extracts 2D patches from a 2D image

    Parameters
    ----------
    image
        2D image

    pixel_spacing
        Spacing between pixels in the image

    pad_value
        Value used to pad extracted patches if they extent beyond the image boundaries

    spline_order
        Order of the spline used for interpolation (if necessary)

    dtype
        Datatype of the extracted patches, theano.config.floatX or 'float32' is usually a good value.
        Defaults to the dtype of the image.
    """

    def __init__(
        self,
        image: np.ndarray,
        pixel_spacing: Optional[Tuple[float, float]] = None,
        pad_value: float = 0,
        spline_order: int = 1,
        dtype: Optional[str] = None,
    ):
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image, got array with {image.ndim} dimensions instead")

        # Add a third dimension to the image
        image_volumized = np.expand_dims(image, axis=2)

        if pixel_spacing is None:
            voxel_spacing = None
        else:
            voxel_spacing = (pixel_spacing[0], pixel_spacing[1], 0.0)

        # Create 3D patch extractor instance
        self.extractor = PatchExtractor3D(
            image_volumized, voxel_spacing, pad_value, spline_order, dtype
        )

    def extract_rect(
        self,
        center_pixel: Tuple[int, int],
        shape: Tuple[int, int],
        extent: Optional[Tuple[float, float]] = None,
        rotation_angle: Optional[float] = None,
    ) -> np.ndarray:
        """Extracts a 2D rectangular patch, see :func:`PatchExtractor3D.extract_rect`"""
        center_voxel = (center_pixel[0], center_pixel[1], 0)
        return self.extractor.extract_rect(
            center_voxel, shape, extent, axis=2, rotation_angle=rotation_angle
        )


class Patch:
    """Represents a patch including the pixel values and the position within the original image space"""

    def __init__(
        self,
        center: Tuple[int, ...],
        array: np.ndarray,
        mask_shape: Tuple[int, ...],
        prediction_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.center = center
        self.array = array

        if prediction_shape is None:
            prediction_shape = array.shape  # assume output size = input size

        self.mask_shape = mask_shape
        self.prediction_shape = prediction_shape
        self.mask_offset, self.prediction_offset = compute_valid_patch_pairs(
            mask_shape, prediction_shape, center
        )

    @property
    def shape(self):
        return self.array.shape

    def copy_prediction_into_mask(
        self, prediction: np.ndarray, mask: np.ndarray, add: bool = False
    ):
        """Copies the network output into a mask of the size of the original image"""
        if mask.shape != self.mask_shape:
            raise ValueError(
                f"Provided mask array has shape {mask.shape}, expected {self.mask_shape}, offset might be wrong"
            )
        if prediction.shape != self.prediction_shape:
            raise ValueError(
                f"Provided prediction array has shape {prediction.shape}, expected {self.prediction_shape}, offset might be wrong"
            )

        if add:
            mask[self.mask_offset] += prediction[self.prediction_offset]
        else:
            mask[self.mask_offset] = prediction[self.prediction_offset]


class _SlidingWindow(IterableDataset):
    def __init__(
        self,
        patch_extractor,
        input_shape,
        output_shape,
        step_size,
        start_with_offset,
        weight_map_sigma_scale,
    ):
        self.patch_extractor = patch_extractor
        self.input_shape = input_shape
        self.output_shape = input_shape if output_shape is None else output_shape
        self.step_size = step_size
        self.start_with_offset = start_with_offset
        self.weight_map_sigma_scale = weight_map_sigma_scale

        # Compute margin relative to the patch center
        self.margin = tuple(m // 2 for m in self.output_shape)

        # Weight map is expensive to compute, so we can best keep it in memory
        self._weight_map = None

    @property
    def image_shape(self) -> Tuple[int, ...]:
        return self.patch_extractor.image.shape

    @property
    def ndim(self) -> int:
        return self.patch_extractor.image.ndim

    @property
    def weight_map(self):
        """Gaussian mask that can be used to weight central pixels stronger than pixels at the boundary"""
        if self._weight_map is None:
            # Create gaussian centered at the patch center
            center_map = np.zeros(self.output_shape)
            center_map[tuple(i // 2 for i in self.output_shape)] = 1
            sigmas = [i * self.weight_map_sigma_scale for i in self.output_shape]
            wm = gaussian_filter(center_map, sigmas, mode="constant", cval=0).astype("float32")

            # Scale weight map to [0, 1] and remove zeros because they might lead to NaNs
            wm = wm / np.max(wm)
            wm[wm == 0] = np.min(wm[wm != 0])

            self._weight_map = wm

        return self._weight_map

    def patch_outside_image(self, center: np.array) -> bool:
        """Checks if a patches output would be completely outside the image"""
        return any(
            c + m < 0 or c - m >= s for c, m, s in zip(center, self.margin, self.image_shape)
        )

    def extract_patch(self, center: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __len__(self) -> int:
        """Number of patches needed to cover the entire image"""
        if self.start_with_offset:
            accessible = self.image_shape
        else:
            accessible = np.asarray(self.image_shape) + self.margin

        return int(
            np.prod(
                [np.ceil(s / (os * self.step_size)) for s, os in zip(accessible, self.output_shape)]
            )
        )

    def __iter__(self) -> Iterator[Patch]:
        """Iterate over scan and yield image patch as well as patch coordinates"""
        ndim = self.ndim
        patch_start = (
            np.array(self.margin) if self.start_with_offset else np.zeros_like(self.input_shape)
        )
        patch_center = patch_start.copy()

        while not self.patch_outside_image(patch_center):
            # Extract image patch
            yield Patch(
                patch_center, self.extract_patch(patch_center), self.image_shape, self.output_shape
            )

            # Move to next patch
            for axis in range(ndim):
                patch_center[axis] += int(round(self.output_shape[axis] * self.step_size))
                if not self.patch_outside_image(patch_center):
                    # patch inside image? we're good, break out of for loop and continue with next iteration
                    break
                elif (axis + 1) < ndim:
                    # patch outside image? reset coordinate and continue moving the patch
                    patch_center[axis] = patch_start[axis]


class SlidingRect(_SlidingWindow):
    """Iterates over a 2D volume and extracts 2D patches in a sliding window fashion"""

    def __init__(
        self,
        patch_extractor: PatchExtractor2D,
        input_shape: Tuple[int, int],
        output_shape: Optional[Tuple[int, int]] = None,
        step_size: float = 1,
        start_with_offset: bool = True,
        weight_map_sigma_scale: float = 0.125,
    ):
        super().__init__(
            patch_extractor,
            input_shape,
            output_shape,
            step_size,
            start_with_offset,
            weight_map_sigma_scale,
        )

    def extract_patch(self, center: np.ndarray) -> np.ndarray:
        return self.patch_extractor.extract_rect(center, shape=self.input_shape)


class SlidingCuboid(_SlidingWindow):
    """Iterates over a 3D volume and extracts 3D patches in a sliding window fashion"""

    def __init__(
        self,
        patch_extractor: PatchExtractor3D,
        input_shape: Tuple[int, int, int],
        output_shape: Optional[Tuple[int, int, int]] = None,
        step_size: float = 1,
        start_with_offset: bool = True,
        weight_map_sigma_scale: float = 0.125,
    ):
        super().__init__(
            patch_extractor,
            input_shape,
            output_shape,
            step_size,
            start_with_offset,
            weight_map_sigma_scale,
        )

    def extract_patch(self, center: np.ndarray) -> np.ndarray:
        return self.patch_extractor.extract_cuboid(center, shape=self.input_shape)
