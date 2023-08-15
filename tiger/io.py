"""
The I/O module provides functions for reading and writing medical images and other data from commonly used
file formats, like .mha/.mhd or .nii.gz or DICOM.

Throughout the library, images are represented as numpy arrays (for the pixel data) with axes ordered x-y-z
and a header object (an instance of the ImageMetadata class) that represents additional metadata, at least
details such as pixel spacing that define the relation of the pixel data to the world coordinate system of
the scanner that acquired the image.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import tifffile as tifio
from panimg.exceptions import UnconsumedFilesException
from panimg.image_builders import image_builder_dicom as panimg_image_builder_dicom

from . import TigerException
from .utils import suppress_output

try:
    import image_loader as diag_image_loader  # custom library for reading (enhanced) DICOM volumes
except ImportError:
    diag_image_loader = None


# We generally except strings and pathlib paths for file and folder names
PathLike = Union[str, Path]


def is_path_like(obj: Any) -> bool:
    return isinstance(obj, str) or isinstance(obj, Path)


def round_almost_integers(values: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    array = np.asarray(values)
    rounded = np.round(array)
    almost_integers = abs(rounded - array) < epsilon
    rounded[~almost_integers] = array[~almost_integers]
    return rounded


# Mapping of metadata names internally, in ITK files and in DICOM headers
class MetadataKey(NamedTuple):
    tiger: str
    itk: str
    dicom: str
    dtype: Callable = str


class ImageMetadata:
    """Wrapper around the metadata of a medical image.

    Can store arbitrary data, but will report default values for three important fields:

    * spacing - defaults to 1mm spacing in all directions
    * origin - defaults to 0
    * direction - defaults to an identity matrix

    Parameters
    ----------
    ndim
        Number of dimensions of the image (>= 2)

    Examples
    --------
    >>> header = ImageMetadata(ndim=3, spacing=(1, 2, 3))
    >>> header["spacing"]
    array([1., 2., 3.])
    """

    special_fields = ("spacing", "origin", "direction")
    known_metadata = (
        MetadataKey("patient_id", itk="PatientID", dicom="0010|0020"),
        MetadataKey("patient_age", itk="PatientAge", dicom="0010|1010"),
        MetadataKey("patient_sex", itk="PatientSex", dicom="0010|0040"),
        MetadataKey("study", itk="StudyInstanceUID", dicom="0020|000d"),
        MetadataKey("series", itk="SeriesInstanceUID", dicom="0020|000e"),
        MetadataKey("modality", itk="Modality", dicom="0008|0060"),
        MetadataKey("image_type", itk="ImageType", dicom="0008|0008"),
        MetadataKey("body_part", itk="BodyPart", dicom="0018|0015"),
        MetadataKey("slice_thickness", itk="SliceThickness", dicom="0018|0050", dtype=float),
        MetadataKey("convolution_kernel", itk="ConvolutionKernel", dicom="0018|1210"),
        MetadataKey("contrast_bolus_agent", itk="ContrastBolusAgent", dicom="0018|0010"),
        MetadataKey(
            "temporal_position_index", itk="TemporalPositionIndex", dicom="0020|9128", dtype=int
        ),
    )
    eps = 0.01

    def __init__(self, ndim: int, **kwargs):
        ndim = int(ndim)
        if not ndim > 1:
            raise ValueError(
                f"Invalid number of dimensions for an image, expected at least 2, got {ndim}"
            )

        self.ndim = ndim
        self.spacing = np.array([1.0] * ndim, dtype=float)
        self.origin = np.array([0.0] * ndim, dtype=float)
        self.direction = np.identity(ndim, dtype=float)

        self.metadata = dict()
        self.update(kwargs)

    @classmethod
    def from_sitk(cls, image: Union[sitk.Image, sitk.ImageFileReader]) -> ImageMetadata:
        """Create ImageMetadata object from SimpleITK image"""
        header = cls(
            ndim=image.GetDimension(),
            spacing=image.GetSpacing(),
            origin=image.GetOrigin(),
            direction=image.GetDirection(),
        )

        for key in cls.known_metadata:
            for metadata_key in (key.itk, key.dicom):
                try:
                    value = str(image.GetMetaData(metadata_key)).strip()
                    if key.dtype != str:
                        value = key.dtype(value)
                    header[key.tiger] = value
                except (ValueError, RuntimeError):
                    continue
                else:
                    break

        return header

    @classmethod
    def from_file(cls, filename: PathLike) -> ImageMetadata:
        """Create ImageMetadata object from file using SimpleITK to read the header only"""
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(filename))
        reader.ReadImageInformation()
        return cls.from_sitk(reader)

    @classmethod
    def from_dict(cls, metadata: Dict[str, Any]) -> ImageMetadata:
        """Create ImageMetadata object from a dictionary"""
        if "ndim" in metadata:
            return cls(**metadata)

        if "spacing" in metadata:
            ndim = len(metadata["spacing"])
        elif "origin" in metadata:
            ndim = len(metadata["origin"])
        elif "direction" in metadata:
            ndim = len(metadata["direction"]) ** (1 / 2)
        else:
            raise ValueError("Could not determine dimensionality of the image")

        return cls(ndim, **metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Export metadata as a dictionary, converting numpy arrays into lists"""
        metadata = {
            "ndim": self.ndim,
            "spacing": self.spacing.astype(float).tolist(),
            "origin": self.origin.astype(float).tolist(),
            "direction": self.direction.flatten().astype(float).tolist(),
        }

        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            else:
                metadata[key] = value

        return metadata

    def __contains__(self, key: str) -> bool:
        return key in self.metadata or key in self.special_fields

    def __getitem__(self, key: str) -> Any:
        if key in self.special_fields:
            return getattr(self, key)
        else:
            return self.metadata[key]

    def __setitem__(self, key: str, value: Any):
        if key == "spacing":
            new_spacing = np.asarray(value, dtype=float).flatten()
            if new_spacing.size == 1:
                new_spacing = np.repeat(new_spacing, self.ndim)
            elif new_spacing.size != self.ndim:
                raise ValueError(
                    f"Pixel spacing of a {self.ndim}-dimensional image can only be a single value (isotropic) or {self.ndim} values"
                )
            self.spacing = new_spacing.copy()
        elif key == "origin":
            new_origin = np.asarray(value, dtype=float).flatten()
            if new_origin.size != self.ndim:
                raise ValueError(
                    f"Expected {self.ndim} values for coordinates of the origin of a {self.ndim}-dimensional image, "
                    f"got {new_origin.size} values"
                )
            self.origin = new_origin.copy()
        elif key == "direction":
            new_direction = np.asarray(value, dtype=float).flatten()

            # Check number of elements
            if new_direction.size != self.ndim**2:
                raise ValueError(
                    f"Expected {self.ndim**2} values for direction cosine matrix of a {self.ndim}-dimensional image, "
                    f"got {new_direction.size} values"
                )
            new_direction = new_direction.reshape((self.ndim, self.ndim))

            # Check if matrix is orthogonal
            if not np.allclose(
                new_direction @ new_direction.T, np.identity(self.ndim), rtol=0, atol=self.eps
            ):
                raise ValueError("Direction matrix needs to be orthogonal")

            self.direction = new_direction.copy()
        else:
            self.metadata[key] = value

    def __delitem__(self, key: str):
        if key in self.special_fields:
            raise KeyError(f'Special field "{key}" cannot be deleted')
        else:
            del self.metadata[key]

    def __eq__(self, other: Any) -> bool:
        """Compares two ImageMetadata instances, returns true if required and additional fields are the same"""
        if not isinstance(other, ImageMetadata):
            return False

        self_metadata = self.metadata.copy()
        if "reader" in self_metadata:
            del self_metadata["reader"]

        other_metadata = other.metadata.copy()
        if "reader" in other_metadata:
            del other_metadata["reader"]

        return (
            np.allclose(self.spacing, other.spacing, rtol=0, atol=self.eps)
            and np.allclose(self.origin, other.origin, rtol=0, atol=self.eps)
            and np.allclose(self.direction, other.direction, rtol=0, atol=self.eps)
            and self_metadata == other_metadata
        )

    def __str__(self) -> str:
        summary = (
            f"ndim={self.ndim}, "
            f"spacing={self.spacing.tolist()}, "
            f"origin={self.origin.tolist()}, "
            f"direction={self.direction.flatten().tolist()}"
        )

        if len(self.metadata) > 0:
            summary += ", " + ", ".join(sorted(self.metadata.keys()))

        return f"ImageMetadata({summary})"

    def __repr__(self) -> str:
        return self.__str__() + super().__repr__()

    def __len__(self) -> int:
        return len(self.special_fields) + len(self.metadata)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def keys(self) -> Iterator[str]:
        yield from self.special_fields
        yield from self.metadata.keys()

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            yield self[key]

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            yield key, self[key]

    def copy(self) -> ImageMetadata:
        """Returns a deep copy of the header"""
        return copy.deepcopy(self)

    def update(self, other: Union[ImageMetadata, Mapping[str, Any]]):
        for key, value in other.items():
            self[key] = value

    def strip(self):
        """Removes additional metadata but retains the coordinate system"""
        self.metadata.clear()

    def has_direction(self, direction: Iterable[float]) -> bool:
        """Tests whether the image's coordinate space has the specified orientation"""
        return bool(
            np.allclose(
                self.direction.flatten(), np.asarray(direction).flatten(), rtol=0, atol=self.eps
            )
        )

    def has_default_direction(self) -> bool:
        """Tests whether the image's coordinate space has the standard orientation (no rotation)"""
        return self.has_direction(np.identity(self.ndim))

    def has_default_orientation(self) -> bool:
        return self.has_default_direction()

    def has_regular_direction(self) -> bool:
        """Tests whether the image's coordinate space has a regular direction (swap/flips of axes, but no rotation)"""
        return bool(
            np.all(
                np.isclose(self.direction, 0, rtol=0, atol=self.eps)
                | np.isclose(abs(self.direction), 1, rtol=0, atol=self.eps)
            )
        )

    def has_regular_orientation(self) -> bool:
        return self.has_regular_direction()

    def world_matrix(self) -> np.ndarray:
        """Combines spacing, origin and direction into a single homogeneous transformation matrix"""
        wm = np.zeros((self.ndim + 1, self.ndim + 1), dtype=float)
        wm[: self.ndim, : self.ndim] = self.direction * self.spacing
        wm[: self.ndim, -1] = self.origin
        wm[-1, -1] = 1
        return wm

    def has_same_world_matrix(self, other: ImageMetadata, epsilon: float = 0.01) -> bool:
        """Compares two headers but only their world matrix, i.e., spacing, origin and direction cosine matrix"""
        return np.allclose(self.world_matrix(), other.world_matrix(), rtol=0, atol=epsilon)

    def physical_coordinates_to_indices(self, physical_coordinates, continuous: bool = False):
        """Converts a vector of world coordinates into pixel coordinates"""
        physical_coordinates = np.asanyarray(physical_coordinates).flatten()
        if physical_coordinates.size != self.ndim:
            raise ValueError(
                f"Expected a coordinate vector with {self.ndim} values, got {physical_coordinates.size} values"
            )

        vector = np.concatenate((physical_coordinates, [1]))
        indices = np.linalg.pinv(self.world_matrix()) @ vector

        return indices[:-1] if continuous else np.around(indices[:-1]).astype(int)

    def indices_to_physical_coordinates(self, indices):
        """Converts a vector of pixel coordinates into world coordinates"""
        indices = np.asanyarray(indices).flatten()
        if indices.size != self.ndim:
            raise ValueError(
                f"Expected an index vector with {self.ndim} values, got {indices.size} values"
            )

        vector = np.concatenate((indices, [1]))
        coordinates = self.world_matrix() @ vector

        return coordinates[:-1]


# --------------------------------------------------------------------------------


# Data type for pixel data plus metadata representation of images
Image = Tuple[np.ndarray, ImageMetadata]


def _reverse_axes(array: np.ndarray) -> np.ndarray:
    return np.transpose(array)


def image_from_sitk(image: sitk.Image) -> Image:
    """Converts a SimpleITK image into a numpy array and corresponding ImageMetadata object"""
    data = _reverse_axes(sitk.GetArrayViewFromImage(image))
    return np.copy(data, order="C"), ImageMetadata.from_sitk(image)


def image_to_sitk(
    data: np.ndarray, header: Optional[ImageMetadata] = None, copy_unknown_metadata: bool = True
) -> sitk.Image:
    """Converts a numpy array, and optionally the corresponding ImageMetadata object, into a SimpleITK image"""
    if data.ndim not in (2, 3, 4):
        raise ValueError(
            f"Cannot convert {data.ndim}D image to SimpleITK Image, only 2D, 3D and 4D are supported"
        )

    if data.ndim == 4:
        # Turn data into series of 3D volumes and combine them into a 4D image
        if header:
            if not np.allclose(
                header["direction"][3, :].flatten(), (0, 0, 0, 1), rtol=0, atol=0.001
            ) or not np.allclose(
                header["direction"][:, 3].flatten(), (0, 0, 0, 1), rtol=0, atol=0.001
            ):
                raise ValueError(
                    "Cannot convert 4D array with rotation in 4th dimension to SimpleITK image"
                )

            header3 = ImageMetadata(
                ndim=3,
                spacing=header["spacing"][:3],
                origin=header["origin"][:3],
                direction=header["direction"][:3, :3],
            )
            spacing4 = header["spacing"][3]
            origin4 = header["origin"][3]
        else:
            header3 = ImageMetadata(ndim=3)
            spacing4 = 1
            origin4 = 0

        image = sitk.JoinSeries(
            [image_to_sitk(data[:, :, :, i], header3) for i in range(data.shape[3])],
            origin4,
            spacing4,
        )
    else:
        image = sitk.GetImageFromArray(_reverse_axes(np.asanyarray(data)))

        if header:
            # Copy standard header (spacing/origin/direction cosine matrix)
            if "spacing" in header:
                image.SetSpacing([float(f) for f in header["spacing"]])
            if "origin" in header:
                image.SetOrigin([float(f) for f in header["origin"]])
            if "direction" in header:
                image.SetDirection([float(f) for f in np.asanyarray(header["direction"]).flatten()])

    if header:
        # Copy additional metadata
        for key in header.known_metadata:
            if key.tiger in header:
                value = str(header[key.tiger]).strip()
                if value != "":
                    image.SetMetaData(key.itk, value)

        if copy_unknown_metadata:
            known_metadata_keys = {"spacing", "origin", "direction"} | {
                key.tiger for key in header.known_metadata
            }
            unknown_metadata_keys = set(header.metadata.keys()) - known_metadata_keys
            for key in unknown_metadata_keys:
                value = str(header[key]).strip()
                if value != "":
                    image.SetMetaData(key, value)

    return image


# --------------------------------------------------------------------------------


class ImageReader(ABC):
    pass


class ImageReaderError(TigerException):
    pass


class ImageReaderNotAvailableError(ImageReaderError):
    def __init__(self, image_reader: ImageReader):
        super().__init__(f'Image reader "{image_reader.__class__.__name__}" is not available')


class ItkImageReader(ImageReader):
    """Reads an image using SimpleITK"""

    def __call__(
        self, filenames: Union[PathLike, Iterable[PathLike]], *, sort: bool = True
    ) -> Image:
        # Make sure we have a list of filenames
        if is_path_like(filenames):
            filenames = [filenames]
        else:
            filenames = list(filenames)

        # Make sure all filenames are Path instances
        filenames = [Path(file) for file in filenames]

        # Check whether all files exist (ITK raises only an unspecific exception)
        if len(filenames) == 0:
            raise ValueError("No files specified")

        for file in filenames:
            if not file.exists():
                raise FileNotFoundError(f'Could not find file "{file}"')

        # Use different approach depending on whether there is only one file or multiple files
        if len(filenames) == 1:
            itk_image = sitk.ReadImage(str(filenames[0]))
            extra_metadata = dict()
        else:
            if sort:
                filenames = sort_dicom_files(filenames)

            # Read image from a set of files
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([str(file) for file in filenames])
            itk_image = reader.Execute()

            # Get additional metadata by reading only one of the files
            extra_metadata = ImageMetadata.from_file(filenames[0]).metadata

        image, header = image_from_sitk(itk_image)
        header.update(extra_metadata)
        header["reader"] = self.__class__.__name__
        return image, header


class NiBabelImageReader(ImageReader):
    """ "Reads a nifti image using NiBabel"""

    def __call__(self, filename: PathLike) -> Image:
        # Read image data
        n = nib.load(str(filename))
        image = np.array(n.dataobj)

        # Convert RAS to LPS
        spacing = n.header.get_zooms()

        ras_to_lps = np.ones((3, 3))
        ras_to_lps[0, :] = -1
        ras_to_lps[1, :] = -1
        direction = round_almost_integers(n.affine[:3, :3] / spacing * ras_to_lps)
        direction[direction == -0.0] = 0

        origin = n.affine[:3, 3] * np.array([-1, -1, 1])

        # Construct header
        header = ImageMetadata(ndim=image.ndim, spacing=spacing, direction=direction, origin=origin)
        header["reader"] = self.__class__.__name__

        return image, header


class TiffImageReader(ImageReader):
    """Reads TIFF images"""

    def __call__(self, filename: PathLike) -> Image:
        data = _reverse_axes(np.asanyarray(tifio.imread(str(filename))))
        return data, ImageMetadata(ndim=data.ndim, reader=self.__class__.__name__)


class TagImageReader(ImageReader):
    """Reads TAG images, a format used by a software called sliceOmatic"""

    DATA_TYPES = {"BYTE": "int8", "SHORT": "int16"}

    def __init__(self):
        self.field_value_pairs = re.compile(r"(\w+):([^\s,]+)")

    def __call__(self, filename: PathLike) -> Image:
        # Read the file into memory
        with Path(filename).open("rb") as fp:
            data = fp.read()

        # Split header from pixel data
        try:
            metadata_str, pixel_data = data.strip().split(b"\x0c", maxsplit=1)
            metadata_str = metadata_str.decode("ASCII")
        except ValueError:
            raise ImageReaderError("Invalid TAG file (header missing)")

        # Parse header
        metadata = dict()
        for line in metadata_str.splitlines():
            # Remove comments
            line = line.split("*", maxsplit=1)[0]

            # Find field-value pairs
            for match in self.field_value_pairs.finditer(line):
                metadata[match.group(1).lower()] = match.group(2)

        # Parse header
        try:
            dtype = self.DATA_TYPES[metadata["type"].upper()]
            shape = [int(metadata[axis]) for axis in ("x", "y", "z")]
            origin = [float(metadata[f"org_{axis}"]) for axis in ("x", "y", "z")]

            spacing = [float(metadata[f"inc_{axis}"]) for axis in ("x", "y")]
            spacing.append(float(metadata["epais"]))

            dir_x = np.array([float(metadata[f"dir_h_{axis}"]) for axis in ("x", "y", "z")])
            dir_y = np.array([float(metadata[f"dir_v_{axis}"]) for axis in ("x", "y", "z")])
            dir_z = np.cross(dir_x, dir_y)
            direction = np.column_stack((dir_x, dir_y, dir_z))
        except KeyError:
            raise ImageReaderError("Invalid TAG file (header invalid)")

        # Assemble metadata object
        header = ImageMetadata(ndim=3, spacing=spacing, origin=origin, direction=direction)
        header["reader"] = self.__class__.__name__

        # Convert pixel data into numpy array
        image = np.frombuffer(pixel_data, dtype=dtype).reshape(shape, order="F")

        # Return a copy of the pixel data because the array is otherwise immutable
        return np.copy(image), header


class DicomReader(ImageReader):
    """Reads DICOM files using the rse-panimg package"""

    def __call__(
        self, filenames: Union[PathLike, Iterable[PathLike]], *, ignore_extra_files: bool = True
    ) -> Iterable[Image]:
        # Make sure that we have a set with filenames and not a path to a folder
        if is_path_like(filenames):
            dcm = Path(filenames)
            if dcm.is_dir():
                dcm_files = {file for file in dcm.glob("*") if file.is_file()}
            elif dcm.suffix == ".zip":
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(dcm, "r") as dcm_folder:
                        dcm_folder.extractall(tmpdir)
                    yield from self(tmpdir, ignore_extra_files=ignore_extra_files)
                return
            else:
                dcm_files = {dcm}
        else:
            dcm_files = {Path(file) for file in filenames}

        # Read all individual series one-by-one
        try:
            for result in panimg_image_builder_dicom(files=dcm_files):
                image, header = image_from_sitk(result.image)

                # Read additional metadata that panimg does not support
                try:
                    extra_metadata = ImageMetadata.from_file(
                        next(iter(result.consumed_files))
                    ).metadata
                    header.update(extra_metadata)
                except RuntimeError:
                    pass  # ITK could not read this file, so no additional metadata

                header["reader"] = self.__class__.__name__
                yield image, header
        except UnconsumedFilesException as e:
            if not ignore_extra_files:
                raise ImageReaderError("Some files do not belong to any DICOM series") from e


class DiagDicomReader(ImageReader):
    """Reads DICOM files using the DIAG image_loader package, if available"""

    def __call__(self, filenames: Union[PathLike, Iterable[PathLike]]) -> Image:
        if diag_image_loader is None:
            raise ImageReaderNotAvailableError(self)

        # Make sure we have a list of filenames
        if is_path_like(filenames):
            filenames = [filenames]
        else:
            filenames = list(filenames)

        # Check whether all files exist (image loader raises only an unspecific exception)
        if len(filenames) == 0:
            raise ValueError("No files specified")

        for file in filenames:
            if not Path(file).exists():
                raise FileNotFoundError(f'Could not find file "{file}"')

        # Make sure all filenames are strings and not Path instances
        filenames = [str(file) for file in filenames]

        # Attempt to load image from DICOM files
        try:
            with suppress_output():
                image = diag_image_loader.load_dicom_image(filenames)
        except diag_image_loader.CppDiagImageLoaderException as e:
            raise ImageReaderError(e.args[0])

        # Convert image and metadata
        data = _reverse_axes(image[0])
        header = ImageMetadata(
            ndim=data.ndim,
            spacing=tuple(reversed(image[3])),
            origin=tuple(reversed(image[2])),
            direction=np.flipud(np.fliplr(np.asarray(image[1]))).T,
            reader=self.__class__.__name__,
        )
        return data, header


def discover_dicom_files(folder: PathLike) -> Mapping[str, List[Path]]:
    """Scans a directory for DICOM files and returns them sorted according to the DICOM coordinate space"""
    reader = sitk.ImageSeriesReader()
    series = dict()
    for series_uid in reader.GetGDCMSeriesIDs(str(folder)):
        filenames = reader.GetGDCMSeriesFileNames(str(folder), series_uid)
        series[series_uid] = [Path(file) for file in filenames]
    return series


def sort_dicom_files(files: Iterable[PathLike]) -> List[Path]:
    """Sorts a list of DICOM files according to the DICOM coordinate space"""
    filenames = list(files)
    if len(filenames) == 0:
        return filenames

    # Determine DICOM directory from filenames
    dcm_folder = filenames[0].parent
    if not all(file.parent == dcm_folder for file in filenames):
        raise ValueError("Cannot sort list of DICOM files that are not in the same folder")

    # Read header of one file to determine series instance UID
    header = ImageMetadata.from_file(filenames[0])
    series_uid = header["series"]

    # Discover DICOM files which will return a sorted list of files
    dcm_files = discover_dicom_files(dcm_folder)
    if series_uid not in dcm_files:
        raise ValueError("Could not find any of the DICOM files that need to be sorted")

    # Filter out files and return sorted list
    basenames = {file.name for file in filenames}
    return [dcm_file for dcm_file in dcm_files[series_uid] if dcm_file.name in basenames]


def read_dicom(filenames: Union[PathLike, Iterable[PathLike]]) -> Image:
    """Reads an image from one or multiple DICOM files

    If there are multiple DICOM series in a specified folder or among the specified files, only on of
    these series is returned. There is no guarantee that a specific series is returned. The alternative
    is using the DicomReader class directly, which gives access to all series.

    Parameters
    ----------
    filenames
        Path to the DICOM file(s), can point to a file or to a directory

    Returns
    -------
    image
        Numpy array containing the pixel data in x-y-z order

    header
        ImageMetadata object containing the header

    Raises
    ------
    ValueError
        If the specified files or folder do not contain a readable DICOM series
    """
    dicom_reader = DicomReader()
    for image_header in dicom_reader(filenames):
        return image_header
    else:
        raise ValueError("Could not find a DICOM series")


def read_image(filename: PathLike) -> Image:
    """Reads an image (2D/3D) in any ITK-supported file format or TIFF (assumed to be grayscale), or DCM.

    Parameters
    ----------
    filename
        Path to the image

    Returns
    -------
    image
        Numpy array containing the pixel data in x-y-z order

    header
        ImageMetadata object containing the header
    """
    filename = Path(filename)
    if filename.suffix in (".tif", ".tiff"):
        image_reader = TiffImageReader()
    elif filename.suffix in (".dcm", ".v2"):
        image_reader = read_dicom
    elif filename.suffix == ".tag":
        image_reader = TagImageReader()
    else:
        image_reader = ItkImageReader()
    return image_reader(filename)


# --------------------------------------------------------------------------------


class ImageWriter(ABC):
    def __init__(
        self,
        *,
        use_compression: bool = True,
        make_dirs: bool = True,
        allow_int64: bool = False,
        strip_metadata: bool = True,
    ):
        self.compression = use_compression
        self.make_dirs = make_dirs
        self.allow_int64 = allow_int64
        self.strip_metadata = strip_metadata

    def _prepare_outdir(self, filename: PathLike):
        outdir = Path(filename).parent
        if self.make_dirs:
            outdir.mkdir(parents=True, exist_ok=True)
        elif not outdir.exists():
            raise OSError("Could not write image, target directory does not exist")

    def _check_int64(self, data: np.ndarray):
        if not self.allow_int64 and data.dtype == np.int64:
            raise ValueError(
                "Image data type is int64, which is not allowed (set allow_int64=True to overwrite this warning)"
            )

    def _strip_metadata(self, header: ImageMetadata) -> ImageMetadata:
        stripped_header = header.copy()

        if self.strip_metadata:
            stripped_header.strip()

        # Always remove the reader name
        try:
            del stripped_header["reader"]
        except KeyError:
            pass

        return stripped_header

    @abstractmethod
    def __call__(
        self,
        filename: PathLike,
        data: np.ndarray,
        header: Optional[ImageMetadata] = None,
    ):
        pass


class ImageWriterError(TigerException):
    pass


class ItkImageWriter(ImageWriter):
    """Writes an image using SimpleITK"""

    def __call__(
        self,
        filename: PathLike,
        data: np.ndarray,
        header: Optional[ImageMetadata] = None,
    ):
        data = np.asarray(data)
        self._check_int64(data)
        header = None if header is None else self._strip_metadata(header)
        sitk_image = image_to_sitk(data, header)
        self._prepare_outdir(filename)
        sitk.WriteImage(sitk_image, str(filename), self.compression)


class TiffImageWriter(ImageWriter):
    """Writes a 2D or 3D grayscale TIFF image"""

    def __call__(
        self,
        filename: PathLike,
        data: np.ndarray,
        header: Optional[ImageMetadata] = None,
    ):
        data = np.asarray(data)
        self._check_int64(data)
        data = _reverse_axes(data)
        if data.ndim not in (2, 3):
            raise ImageWriterError(f"Writing of {data.ndim}D tif files is not supported")

        kwargs = {
            "photometric": "minisblack",
            "compression": "ADOBE_DEFLATE" if self.compression else 0,
        }
        if data.ndim == 3:
            if data.shape[1] % 16 != 0 or data.shape[2] % 16 != 0:
                raise ImageWriterError(
                    "Writing 3D tif files is only supported for images with x,y dimensions that are multiples of 16"
                )
            kwargs["tile"] = (1,) + data.shape[1:]

        self._prepare_outdir(filename)
        tifio.imwrite(str(filename), np.ascontiguousarray(data), **kwargs)


def write_image(
    filename: PathLike,
    data: np.ndarray,
    header: Optional[ImageMetadata] = None,
    *,
    use_compression: bool = True,
    make_dirs: bool = True,
    allow_int64: bool = False,
    strip_metadata: bool = True,
):
    """Saves an image (2D/3D) in any ITK-supported file format or as TIFF grayscale file.

    Parameters
    ----------
    filename
        Path to the image. The file extension determines the file format.

    data
        Numpy array containing the pixel data in x-y-z order

    header
        Metadata for the image, such as spacing or origin of the image coordinate system.

    use_compression
        Whether the image should be saved in compressed form if supported by the file format.

    make_dirs
        Whether the function makes sure that the parent directory exists before creating the
        new image file.

    allow_int64
        Whether data with dtype int64 will lead to a ValueError or whether it will be accepted.
        The reason for rejecting this kind of data by default are problems with loading this
        data in Mevislab, so that it is often not desired to store e.g. masks with this dtype
        even though numpy uses this as default integer type.

    strip_metadata
        Whether additional (potentially privacy sensitive) metadata is stripped from the header
        before saving the file.
    """
    kwargs = {
        "use_compression": use_compression,
        "make_dirs": make_dirs,
        "allow_int64": allow_int64,
        "strip_metadata": strip_metadata,
    }

    imagefile = Path(filename)
    if imagefile.suffix in (".tif", ".tiff"):
        image_writer = TiffImageWriter(**kwargs)
    else:
        image_writer = ItkImageWriter(**kwargs)
    image_writer(filename, data, header)


def convert_image(input_filename: PathLike, output_filename: PathLike, *, dtype=None, **kwargs):
    """Reads an image from one file and saves it to another"""
    data, header = read_image(input_filename)
    if dtype:
        data = data.astype(dtype)
    write_image(output_filename, data, header, **kwargs)


# --------------------------------------------------------------------------------


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return o.as_posix()
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


def read_json(filename: PathLike, *, ordered_dict: bool = True, **kwargs):
    """Reads a json file"""
    if ordered_dict:
        kwargs["object_pairs_hook"] = OrderedDict

    with Path(filename).open() as fp:
        return json.load(fp, **kwargs)


def write_json(
    filename: PathLike,
    data: Any,
    *,
    encoding: str = "UTF-8",
    make_dirs: bool = True,
    **kwargs,
):
    """Dumps an object into a json file, using pretty printing and UTF-8 encoding by default"""
    jsonfile = Path(filename)

    # Ensure that the directory exists
    if make_dirs:
        jsonfile.parent.mkdir(parents=True, exist_ok=True)

    # Write data into json file
    args = {"sort_keys": False, "indent": 2, "cls": JSONEncoder}
    args.update(kwargs)

    with jsonfile.open("w", encoding=encoding) as fp:
        json.dump(data, fp, **args)


# --------------------------------------------------------------------------------


def refresh_file_list(path: PathLike):
    """Update the cached file list of directories on a network share

    On network shares (chansey in particular), files are sometimes reported missing even though they
    exist. This has to do with caching issues and can be fixed by running "ls" or similar commands
    in the parent directory of these files.
    """
    path = Path(path)  # make sure path is a OS-specific path object
    if sys.platform == "win32":
        cmd = ["cmd", "/c", "dir", str(path)]
    else:
        cmd = ["ls", str(path)]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise OSError(f'Could not refresh file list of directory "{path}"') from e


def path_exists(path: PathLike) -> bool:
    """Checks whether the file or directory exists

    Unlike checks via os.path or pathlib, this check works reliably also on network shares where the
    content of directories might be cached.
    """
    path = Path(path)

    # Refresh content of parent (folder in which the object of interest is stored)
    try:
        refresh_file_list(path.parent)
    except OSError:
        # If the parent directory does not exist, refreshing the file list will fail, but that's okay
        pass

    # Now we can check if the specific file/directory exists
    if path.exists():
        # If the object was a directory itself, we also ask for a fresh list of it's content
        if path.is_dir() and path != path.parent:
            refresh_file_list(path)
        return True
    else:
        return False


def checksum(file: PathLike, algorithm: str = "sha256", chunk_size: int = 4096) -> str:
    """Computes the checksum of a file using a hashing algorithm"""
    file = Path(file)
    if file.exists() and not file.is_file():
        raise ValueError(f"Checksum can be computed only for files, {file} is not a file")

    h = hashlib.new(algorithm)
    with file.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def copyfile(src: PathLike, dst: PathLike, **kwargs) -> Path:
    """Similar to shutil.copyfile but accepts a directory as input for dst"""
    src = Path(src)
    dst = Path(dst)

    if dst.is_dir():
        dst /= src.name

    return shutil.copyfile(src, dst, **kwargs)


def copytree(src: PathLike, dst: PathLike, ignore=None) -> Path:
    """Similar to shutil.copytree but makes sure that copyfile is used for copying"""
    try:
        shutil.copytree(
            src,
            dst,
            ignore=ignore,
            symlinks=False,
            ignore_dangling_symlinks=True,
            copy_function=copyfile,
        )
    except shutil.Error as e:
        non_permission_errors = []
        for error in e.args[0]:
            msg = error[2] if isinstance(error, tuple) else error
            if "Operation not permitted" not in msg:
                non_permission_errors.append(error)

        if len(non_permission_errors) > 0:
            raise shutil.Error(non_permission_errors)

    return Path(dst)
