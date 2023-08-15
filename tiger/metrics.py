from typing import Iterable, Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as connected_components


class BlandAltmanLimitsOfAgreement:
    """Computes Bland-Altman limits of agreement (95% confidence interval) for two sets of measurements

    Parameters
    ----------
    manual
        Manual (reference) scores
    automatic
        Auomatically computed (test) scores
    """

    def __init__(self, manual: Iterable[float], automatic: Iterable[float]):
        # Try to turn input variables into numpy arrays
        scores: np.ndarray = np.asarray(automatic)
        reference_scores = np.asarray(manual)

        # Compute limits of agreement
        self.diffs = reference_scores - scores
        self.means = (reference_scores + scores) / 2.0

        self.mean_diff = np.mean(self.diffs)  # Mean difference of all measurements (offset)

        std_diff = np.std(self.diffs)
        self.upper_limit = self.mean_diff + 1.96 * std_diff  # Lower 95% limit of agreement
        self.lower_limit = self.mean_diff - 1.96 * std_diff  # Upper 95% limit of agreement


def jaccard_score(mask1: Iterable[bool], mask2: Iterable[bool]) -> float:
    """Jaccard volume overlap score for two binary masks"""
    m1 = np.asarray(mask1, dtype="bool").flatten()
    m2 = np.asarray(mask2, dtype="bool").flatten()

    try:
        return np.count_nonzero(m1 & m2) / float(np.count_nonzero(m1 | m2))
    except ZeroDivisionError:
        raise ValueError("Cannot compute dice score on empty masks")


def dice_score(mask1: Iterable[bool], mask2: Iterable[bool]) -> float:
    """Dice volume overlap score for two binary masks"""
    m1 = np.asarray(mask1, dtype="bool").flatten()
    m2 = np.asarray(mask2, dtype="bool").flatten()

    try:
        return 2 * np.count_nonzero(m1 & m2) / float(np.count_nonzero(m1) + np.count_nonzero(m2))
    except ZeroDivisionError:
        raise ValueError("Cannot compute dice score on empty masks")


def mean_dice_score(mask1: Iterable[int], mask2: Iterable[int], labels: Iterable[int]) -> float:
    """Dice volume overlap score for two multi-label masks, averaged over all labels"""
    m1 = np.asarray(mask1)
    m2 = np.asarray(mask2)

    labelwise_scores = []
    for label in labels:
        labelwise_scores.append(dice_score(m1 == label, m2 == label))

    if len(labelwise_scores) == 0:
        raise ValueError("Cannot compute dice score on empty masks")

    return float(np.mean(labelwise_scores))


def surface_distances(
    manual: Iterable[bool],
    automatic: Iterable[bool],
    voxel_spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
) -> Iterable[float]:
    """Computes the surface distances (positive numbers) from all border voxels of a binary object in two images."""
    manual_mask = np.asarray(manual, dtype="bool")
    automatic_mask = np.asarray(automatic, dtype="bool")

    if np.count_nonzero(manual_mask) == 0 or np.count_nonzero(automatic_mask) == 0:
        raise ValueError(
            "Cannot compute surface distance if there are no foreground pixels in the image"
        )

    if connectivity is None:
        connectivity = manual_mask.ndim

    # Extract border using erosion
    footprint = generate_binary_structure(manual_mask.ndim, connectivity)
    manual_border = manual_mask ^ binary_erosion(manual_mask, structure=footprint, iterations=1)
    automatic_border = automatic_mask ^ binary_erosion(
        automatic_mask, structure=footprint, iterations=1
    )

    # Compute average surface distance
    dt = distance_transform_edt(~manual_border, sampling=voxel_spacing)
    return dt[automatic_border]


def hausdorff_distance(
    manual: Iterable[bool],
    automatic: Iterable[bool],
    voxel_spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
    symmetric: bool = True,
) -> float:
    """
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two images.

    Parameters
    ----------
    manual
        Reference masks (binary)

    automatic
        Masks that is compared to the reference mask

    voxel_spacing
        Spacing between elements in the images

    connectivity
        The neighbourhood/connectivity considered when determining the surface of the binary objects. Values between 1 and ndim are valid.
        Defaults to ndim, which is full connectivity even along the diagonal.

    symmetric
        Whether the distance is calculated from manual to automatic mask, or symmetrically (max distance in either direction)

    Returns
    -------
    float
        Hausdorff distance
    """
    hd1 = max(surface_distances(manual, automatic, voxel_spacing, connectivity))
    if not symmetric:
        return hd1

    hd2 = max(surface_distances(automatic, manual, voxel_spacing, connectivity))
    return float(max(hd1, hd2))


def average_surface_distance(
    manual: Iterable[bool],
    automatic: Iterable[bool],
    voxel_spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
    symmetric: bool = True,
) -> float:
    """
    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    manual
        Reference masks (binary)

    automatic
        Masks that is compared to the reference mask

    voxel_spacing
        Spacing between elements in the images

    connectivity
        The neighbourhood/connectivity considered when determining the surface of the binary objects. Values between 1 and ndim are valid.
        Defaults to ndim, which is full connectivity even along the diagonal.

    symmetric
        Whether the surface distance are calculated from manual to automatic mask, or symmetrically in both directions

    Returns
    -------
    float
        Average surface distance
    """
    sd1 = surface_distances(manual, automatic, voxel_spacing, connectivity)
    if not symmetric:
        return np.asarray(sd1).mean()

    sd2 = surface_distances(automatic, manual, voxel_spacing, connectivity)
    return float(np.concatenate((sd1, sd2)).mean())


def agatston_score(
    image: np.ndarray,
    mask: np.ndarray,
    voxel_spacing: Iterable[float],
    slice_thickness: float = 3.0,
    correct_spacing: bool = True,
    correct_thickness: bool = False,
    threshold: bool = True,
) -> int:
    """
    Calculates the total Agatston score

    Optionally corrects for slice spacing and slice thickness other than 3mm. Input mask is expected to be binary.

    Parameters
    ----------
    image
        The image as a numpy array, should be a 3D volume
    mask
        The calcium mask (binary) with the same shape as the image
    voxel_spacing
        The spacing between voxels in the image
    slice_thickness
        The slice thickness of the CT scan (used to correct for slice thickness != 3 mm)
    correct_spacing
        Apply a correction factor if slice spacing != 3 mm?
    correct_thickness
        Apply a correction factor if slice thickness != 3 mm?
    threshold
        Remove calcium voxels below the 130 HU threshold?

    Returns
    -------
    int
        The Agatston score, rounded to the next integer
    """
    image = np.atleast_3d(image)
    mask = np.atleast_3d(mask).astype(bool)
    voxel_spacing = tuple(voxel_spacing)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask need to have the same shape, got {image.shape} and {mask.shape}"
        )

    if threshold:
        mask = np.logical_and(mask, image >= 130)

    score = 0
    connectivity = generate_binary_structure(2, 2)
    for z in range(image.shape[2]):
        labels, n_components = connected_components(mask[:, :, z], connectivity)
        for component_label in range(1, n_components + 1):
            component = labels == component_label
            n_pixels = np.count_nonzero(component)

            # Determine coefficient from maximum intensity in the component in this slice
            maximum_intensity = np.max(image[:, :, z][component])
            if maximum_intensity < 200:
                coefficient = 1
            elif maximum_intensity < 300:
                coefficient = 2
            elif maximum_intensity < 400:
                coefficient = 3
            else:
                coefficient = 4

            score += coefficient * n_pixels

    pixel_area = voxel_spacing[0] * voxel_spacing[1]
    spacing_factor = voxel_spacing[2] / 3.0 if correct_spacing else 1
    thickness_factor = slice_thickness / 3.0 if correct_thickness else 1
    correction_factor = spacing_factor * thickness_factor

    return int(round(score * pixel_area * correction_factor))


def volume_score(mask: np.ndarray, voxel_spacing: Iterable[float]) -> float:
    """Computes the total volume of the positive voxels in the mask"""
    n_voxels = np.count_nonzero(mask)
    voxel_volume = np.prod(np.asarray(voxel_spacing, dtype=float))
    return float(n_voxels * voxel_volume)


def mass_score(image: np.ndarray, mask: np.ndarray, voxel_spacing: Iterable[float]) -> float:
    """Computes the total pseudo mass score (in mg) of the positive voxels in the mask"""
    if image.shape != mask.shape:
        raise ValueError(
            f"Image and mask need to have the same shape, got {image.shape} and {mask.shape}"
        )

    calcium_voxels = mask > 0
    voxel_volume = np.prod(voxel_spacing)
    sum_intensity = np.sum(image[calcium_voxels])
    calcium_mass_in_g = voxel_volume * sum_intensity
    return 0.001 * calcium_mass_in_g  # in mg
