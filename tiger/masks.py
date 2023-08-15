"""
Manipulation of masks (binary or multiclass)
"""

from typing import Any, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from skimage.measure import label as connected_components

from . import TigerException
from .io import Image, ImageMetadata
from .patches import Patch
from .utils import slice_nd_array


class BoundingBox:
    """Bounding box of the object(s) in a binary mask"""

    def __init__(self, mask: Union[np.ndarray, Patch]):
        if isinstance(mask, Patch):
            self.shape = mask.shape
            self.mask_shape = mask.mask_shape
            self.lower_corner = np.array(
                [max(0, c - s // 2) for c, s in zip(mask.center, mask.shape)]
            )
            self.upper_corner = np.array(
                [min(c + s // 2, ss) for c, s, ss in zip(mask.center, mask.shape, mask.mask_shape)]
            )
        else:
            # Store original mask shape
            mask = np.asarray(mask)
            self.mask_shape = mask.shape

            # Find corners
            corners = BoundingBox.find_corners(mask)
            if corners is None:
                self.lower_corner = None
                self.upper_corner = None
                self.shape = None
            else:
                self.lower_corner, self.upper_corner = corners
                self.shape = tuple(self.upper_corner - self.lower_corner + 1)

    @property
    def dims(self) -> int:
        """Dimensionality of the bounding box (e.g. 3 for 3D)"""
        return len(self.mask_shape)

    @property
    def empty(self) -> bool:
        """An empty mask will lead to an empty (=invalid) bounding box"""
        return self.shape is None

    @property
    def center(self) -> Optional[np.ndarray]:
        """Either the coordinates of the center of the bounding box, or None for an empty bounding box"""
        if self.empty:
            return None
        else:
            return 0.5 * (self.lower_corner + self.upper_corner)

    @property
    def size(self) -> int:
        """Size (number of voxels) of the bounding box"""
        if self.empty:
            return 0
        else:
            return int(np.prod(self.shape))

    def __len__(self) -> int:
        return self.size

    def contains(self, p: Iterable[float]) -> bool:
        """Checks whether the specified point lies within the bounding box"""
        if self.empty:
            return False

        coords = np.asarray(p).flatten()
        if len(coords) != self.dims:
            raise ValueError(
                f"Expected voxel coordinates with {self.dims} dimensions, got {coords}"
            )

        return (
            np.count_nonzero(np.logical_or(coords < self.lower_corner, coords > self.upper_corner))
            == 0
        )

    def __contains__(self, p: Iterable[float]) -> bool:
        return self.contains(p)

    def make_mask(self) -> np.ndarray:
        """Creates a mask of the same size as the original mask with the bounding box set to 1 and the rest set to 0"""
        m = np.zeros(shape=self.mask_shape, dtype=bool)
        if not self.empty:
            m[
                tuple(
                    slice(lower, upper + 1)
                    for lower, upper in zip(self.lower_corner, self.upper_corner)
                )
            ] = True
        return m

    def crop(self, image: np.ndarray, copy: bool = True) -> np.ndarray:
        """Crops an image of the same size as the original mask to the bounding box"""
        if image.shape != self.mask_shape:
            raise ValueError(
                f"Expected array with shape {self.mask_shape}, got shape {image.shape}"
            )

        if self.empty:
            return np.empty([0] * self.dims, dtype=image.dtype)

        cropped = image[
            tuple(
                slice(lower, upper + 1)
                for lower, upper in zip(self.lower_corner, self.upper_corner)
            )
        ]
        return np.copy(cropped) if copy else cropped

    @staticmethod
    def find_corners(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns the lower and upper corner of the bounding box of all nonzero elements in the provided mask"""
        indices = np.nonzero(mask)
        if len(indices[0]) == 0:
            return None  # no foreground voxels

        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        return np.array(lower), np.array(upper)


def retain_largest_components(
    mask: np.ndarray,
    labels: Optional[Iterable[int]] = None,
    n: int = 1,
    background: int = 0,
    connectivity: Optional[int] = None,
) -> np.ndarray:
    """
    Returns a new numpy array with only the n largest connected components retained per label.

    Parameters
    ----------
    mask
        Numpy array with integer labels

    labels
        List of labels to retain. If not provided, the mask is first searched for unique values.

    n
        Number of components to retain per label

    background
        Background value

    connectivity
        Determines the connectivity that defines connected-ness. Values between 1 and mask.ndim are
        permitted, see manual of skimage.measure.label for details. Defaults to full connectivity.

    Returns
    -------
    mask
        Numpy array with the same shape and dtype as the input mask
    """
    # Determine labels present in the mask if a list of labels was not provided
    if labels is None:
        labels = np.unique(mask)

    reduced_mask = np.full_like(mask, fill_value=background)
    for label in labels:
        if label == background:
            continue

        cmap = connected_components(mask == label, connectivity=connectivity)
        components = np.unique(cmap[cmap > 0], return_counts=True)

        for i, component in enumerate(sorted(zip(*components), key=lambda c: c[1], reverse=True)):
            if i >= n:
                break
            reduced_mask[cmap == component[0]] = label

    return reduced_mask


class ConnectedComponent(NamedTuple):
    """Represents a connected component in a mask"""

    index: int
    label: Any
    size: int  # number of voxels
    mask: np.ndarray  # binary mask of the connected component


class ConnectedComponents:
    """Finds connected components in a mask"""

    def __init__(self, mask: np.ndarray, background: int = 0, connectivity: Optional[int] = None):
        self.map, n_components = connected_components(
            mask, background=background, return_num=True, connectivity=connectivity
        )

        # Assemble list of components
        components = []
        mask_flattened = mask.flatten()
        for index in range(1, n_components + 1):
            c = self.map == index
            components.append(
                ConnectedComponent(
                    index=index,
                    label=mask_flattened[np.argmax(c)].item(),
                    size=np.count_nonzero(c),
                    mask=c,
                )
            )

        # Sort list of components by size, from small to large
        self.components: List[ConnectedComponent] = list(
            sorted(
                components, key=lambda component: (component.size, component.label, component.index)
            )
        )

    def __len__(self) -> int:
        """Number of components"""
        return len(self.components)

    def __getitem__(self, index: int) -> ConnectedComponent:
        """Component at index, where lower indices correspond to smaller components"""
        return self.components[index]

    def __iter__(self) -> Iterator[ConnectedComponent]:
        """All components, from small to large"""
        yield from self.components

    def filter(self, label: Any) -> Iterator[ConnectedComponent]:
        """Yields only components with the specified label, from small to large"""
        for component in self.components:
            if component.label == label:
                yield component

    def labels(self) -> List[Any]:
        """Returns a list of labels for which components were found"""
        return list(sorted({component.label for component in self.components}))

    @property
    def largest(self) -> Optional[ConnectedComponent]:
        """Returns the largest component"""
        try:
            return self.components[-1]
        except IndexError:
            return None

    @property
    def smallest(self) -> Optional[ConnectedComponent]:
        """Returns the smallest component"""
        try:
            return self.components[0]
        except IndexError:
            return None


def most_common_labels(mask: Iterable[int], background: int = 0):
    """Returns a list of the labels in the mask, sorted by frequency"""
    labels, counts = np.unique(mask, return_counts=True)
    sorted_labels = sorted(zip(labels, counts), key=lambda p: p[1], reverse=True)
    return [p[0] for p in sorted_labels if p[0] != background]


class OverlappingMasksError(TigerException):
    """Raised when masks are overlapping in situations where this is not allowed"""


def _crop_mask_in_image(
    mask: np.ndarray,
    max_shape: Iterable[int],
    mask_offset: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Crop masks within the boundaries of the reference image.

    Parameters
    ----------
    mask
        Mask to crop
    max_shape
        Maximum shape of the reference image.
    mask_offset
        Offset in dimensions of the mask.

    Returns
    -------
    cropped_mask
        Mask cropped to specified shape
    """

    ndim = len(mask.shape)

    if mask_offset is None:
        mask_offset = [0] * ndim
    else:
        mask_offset = list(mask_offset)

    # Crop the mask to make sure it fits in the required shape.
    return mask[
        tuple(
            np.s_[max(0, -o) : min(s, ms - o)]
            for s, ms, o in zip(mask.shape, max_shape, mask_offset)
        )
    ]


def merge_masks(
    reference_image_shape: Iterable[int],
    reference_image_header: ImageMetadata,
    masks: Iterable[Image],
    unique_labels: bool = True,
    binarize: bool = True,
    strict: bool = True,
    dtype: Optional[Union[str, np.dtype]] = None,
) -> np.ndarray:
    """Merge multiple smaller masks into one big mask of a specified shape.

    Relies on the physical coordinates of the individual masks to find their place in the larger mask.

    Parameters
    ----------
    reference_image_shape
        Shape of the reference image.
    reference_image_header
        Header belonging to the reference image.
    masks
        List of masks (data, header) tuples to be merged.
    unique_labels
        If true, every mask gets a unique label, otherwise the output is binary.
    binarize
        If true, every mask is binarized before it is being merged.
    strict
        Forbid overlapping masks.
    dtype
        Data type of the returned mask, is otherwise inferred from the given masks.

    Returns
    -------
    mask
        Merged mask.

    Raises
    ------
    ValueError
        If the masks do not have the same spacing and orientation
    OverlappingMasksError
        If strict=True this exception is raised if masks overlap
    """

    # Create an empty mask of the target shape
    if dtype is None:
        for mask, _ in masks:
            dtype = mask.dtype
            break

    merged_mask = np.zeros(tuple(reference_image_shape), dtype=dtype)

    # Simply use the same header for the requested merged mask as it should be
    # used as an overlay.
    merged_mask_header = reference_image_header

    # Check whether all masks have the same spacing and direction, otherwise merging is more complex
    for _, mask_header in masks:
        for field in ("spacing", "direction"):
            if not np.allclose(mask_header[field], merged_mask_header[field], rtol=0, atol=0.001):
                raise ValueError(
                    f"Only masks with the same {field} can be merged, expected {merged_mask_header[field]}, got {mask_header[field]}"
                )

    # Add every small mask to the merged mask.
    for i, (mask, mask_header) in enumerate(masks):
        single_mask = mask.copy()

        # Retrieve indices from the offset.
        offset_physical = mask_header["origin"]

        # Get indexed offset with respect to the original image.
        offset_indices = reference_image_header.physical_coordinates_to_indices(offset_physical)

        # Crop the mask to not place it somewhere out of bounds.
        single_mask = _crop_mask_in_image(
            mask=single_mask, max_shape=merged_mask.shape, mask_offset=offset_indices
        )

        # Only now remove negative indices as they are used for cropping.
        offset_indices = tuple(max(0, i) for i in offset_indices)

        if binarize:
            single_mask = (single_mask > 0).astype(dtype)

        # Get the annotation data and assign it a label (the maximum unused value).
        if unique_labels:
            single_mask[np.nonzero(single_mask)] = np.max(merged_mask) + 1

        mask_slice = slice_nd_array(slice_shape=single_mask.shape, offset=offset_indices)

        if strict:
            # Check if the new annotation overlaps with a previous annotation.
            if np.any(merged_mask[mask_slice] * single_mask):
                raise OverlappingMasksError(f"Submask {i} overlaps with a previously added mask.")

        # Add the annotation to the merged_mask. Keep the existing annotations
        # there, so only past on zero entries.
        merged_mask[mask_slice] = np.where(
            merged_mask[mask_slice] != 0, merged_mask[mask_slice], single_mask
        )

    return merged_mask
