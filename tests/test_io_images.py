import shutil
import zipfile

import numpy as np
import pytest
import SimpleITK as sitk

import tiger.io


@pytest.fixture
def numpy_array():
    array = np.zeros(shape=(32, 16, 8), dtype="int16")
    array[8, 2, 0] = 1
    array[4, 11, 7] = -1
    return array


@pytest.fixture
def numpy_array2():
    array = np.zeros(shape=(32, 24), dtype="float32")
    array[8, 2] = 1
    array[4, 11] = -1
    return array


@pytest.fixture
def numpy_array4():
    array = np.zeros(shape=(32, 24, 16, 8), dtype="uint8")
    array[8, 2, 3, 4] = 1
    array[4, 11, 0, 1] = 2
    return array


@pytest.fixture
def sitk_image():
    image = sitk.Image(32, 16, 8, sitk.sitkInt16)
    image.SetPixel(8, 2, 0, 1)
    image.SetPixel(4, 11, 7, -1)
    image.SetSpacing((1, 2.5, 3))
    image.SetOrigin((0, 1.5, 2))
    image.SetDirection((0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


@pytest.fixture
def header():
    return tiger.io.ImageMetadata(
        ndim=3,
        spacing=(1, 2.5, 3),
        origin=(0, 1.5, 2),
        direction=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )


@pytest.fixture
def header2():
    return tiger.io.ImageMetadata(
        ndim=2, spacing=(1, 2.5), origin=(0, 1.5), direction=(1.0, 0.0, 0.0, 1.0)
    )


@pytest.fixture
def header4():
    return tiger.io.ImageMetadata(
        ndim=4,
        spacing=(1, 1, 0.5, 1.75),
        origin=(-400, 120, 1.5, 69.25),
        direction=(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1),
    )


def test_sitk_to_numpy_with_header(sitk_image, numpy_array, header):
    """Test conversion of image and extraction of header"""
    arr, h = tiger.io.image_from_sitk(sitk_image)

    assert np.array_equal(arr, numpy_array)
    assert h == header


def test_numpy_to_sitk_data_only(numpy_array, sitk_image):
    """Test converion from numpy to simple itk image"""
    img = tiger.io.image_to_sitk(numpy_array)

    # Transfer metadata to make sure that a comparison of the images results in an error if the actual data is different
    img.CopyInformation(sitk_image)
    assert img == sitk_image


def test_numpy_to_sitk_with_header(numpy_array, header, sitk_image):
    """Test converion from numpy + header to simple itk image"""
    img = tiger.io.image_to_sitk(numpy_array, header=header)
    assert img == sitk_image


def test_numpy_to_sitk_4d(numpy_array4, header4):
    img = tiger.io.image_to_sitk(numpy_array4, header4)
    arr, h = tiger.io.image_from_sitk(img)
    assert np.array_equal(arr, numpy_array4)
    assert h == header4

    # Modify direction vector to introduce a rotation in the 4th axis (not supported currently)
    header4_different_direction = header4.copy()
    header4_different_direction["direction"] = (0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError):
        tiger.io.image_to_sitk(numpy_array4, header4_different_direction)


def test_write_into_nonexisting_directory(numpy_array, tmp_path):
    """Writing should fail if directory does not exist and make_dirs=False"""
    filename = tmp_path / "io" / "image.mha"
    with pytest.raises(IOError):
        tiger.io.write_image(filename, numpy_array, make_dirs=False)


def test_read_from_nonexisting_file(tmp_path):
    """Reading from non-existing file should result in an IOError"""
    filename = tmp_path / "does_not_exist.mha"
    with pytest.raises(IOError):
        tiger.io.read_image(filename)


def test_write_without_compression(numpy_array, tmp_path):
    """Should produce raw file for mhd files without compression"""
    filename = tmp_path / "io" / "image.mhd"
    tiger.io.write_image(filename, numpy_array, use_compression=False)
    assert filename.with_suffix(".raw").exists()
    shutil.rmtree(filename.parent)


def test_write_with_compression(numpy_array, tmp_path):
    """Should produce zraw file for mhd files with compression"""
    filename = tmp_path / "io" / "image.mhd"
    tiger.io.write_image(filename, numpy_array, use_compression=True)
    assert filename.with_suffix(".zraw").exists()
    shutil.rmtree(filename.parent)


@pytest.mark.parametrize("filetype", ["mha", "tif"])
@pytest.mark.parametrize("compression", [True, False])
def test_read_write(filetype, compression, numpy_array, header, tmp_path):
    """Reading from the just written file should reproduce content and header"""
    filename = tmp_path / "io" / f"image.{filetype}"

    # Write to file
    tiger.io.write_image(filename, numpy_array, header, use_compression=compression)
    assert filename.exists()

    # Read back
    arr, h = tiger.io.read_image(filename)

    # Compare pixel data
    assert np.array_equal(arr, numpy_array)
    assert arr.dtype == numpy_array.dtype

    # Compare header
    if filetype == "tif":
        assert h == tiger.io.ImageMetadata(
            ndim=numpy_array.ndim
        )  # tif files do not contain metadata
    else:
        assert h == header

    # Cleanup
    shutil.rmtree(filename.parent)


@pytest.mark.parametrize("filetype", ["mha", "tif"])
@pytest.mark.parametrize("compression", [True, False])
def test_read_write_2d(filetype, compression, numpy_array2, header2, tmp_path):
    """Reading from the just written file should reproduce content and header (2D images)"""
    test_read_write(filetype, compression, numpy_array2, header2, tmp_path)


@pytest.mark.parametrize("filetype", ["mha", "tif"])
@pytest.mark.parametrize("compression", [True, False])
def test_read_write_4d(filetype, compression, numpy_array4, header4, tmp_path):
    """Reading from the just written file should reproduce content and header (2D images)"""
    if filetype == "tif":
        with pytest.raises(
            tiger.io.ImageWriterError
        ):  # 4D tif is currently not supported, so should raise an error
            test_read_write(filetype, compression, numpy_array4, header4, tmp_path)
    else:
        test_read_write(filetype, compression, numpy_array4, header4, tmp_path)


@pytest.mark.parametrize("filetype", ["mha", "mhd"])
def test_read_write_metadata(filetype, numpy_array, header, tmp_path):
    """Writing metadata as part of the header and reading it back from the same file"""
    filename = tmp_path / "io" / f"image.{filetype}"

    # Add some metadata
    header["slice_thickness"] = 1.5
    header["convolution_kernel"] = "FC51"

    # Write to file and read back
    tiger.io.write_image(filename, numpy_array, header, strip_metadata=False)
    arr, h = tiger.io.read_image(filename)

    assert h["slice_thickness"] == header["slice_thickness"]
    assert h["convolution_kernel"] == header["convolution_kernel"]

    # Cleanup
    shutil.rmtree(filename.parent)


def test_strip_metadata(numpy_array, header, tmp_path):
    """Writing metadata as part of the header and reading it back from the same file"""
    filename = tmp_path / "io" / "image.mha"

    # Add some metadata
    header["slice_thickness"] = 1.5

    # Write to file and read back
    tiger.io.write_image(filename, numpy_array, header, strip_metadata=True)
    arr, h = tiger.io.read_image(filename)

    assert "slice_thickness" not in h

    # Cleanup
    shutil.rmtree(filename.parent)


@pytest.mark.parametrize("strip", [True, False])
def test_strip_reader_name(numpy_array, header, strip, tmp_path):
    # Default header has no reader entry
    assert "reader" not in header

    # Reading the same image from a file gives us a reader entry
    f = tmp_path / "io" / "image.mha"
    tiger.io.write_image(f, numpy_array, header, strip_metadata=strip)
    arr, h = tiger.io.read_image(f)
    assert "reader" in h

    # This reader entry is actually not stored in the file though
    h2 = tiger.io.ImageMetadata.from_file(f)
    assert "reader" not in h2


def test_read_write_str(numpy_array, header, tmp_path):
    """Write and read using strings as filenames"""
    imagefile = tmp_path / "io" / "image.mha"
    assert not imagefile.exists()

    # Write to file and read back
    tiger.io.write_image(str(imagefile), numpy_array, header)
    assert imagefile.exists()

    arr, h = tiger.io.read_image(str(imagefile))
    assert np.array_equal(arr, numpy_array)

    # Cleanup
    shutil.rmtree(imagefile.parent)


def test_write_int64(numpy_array, header, tmp_path):
    imagefile = tmp_path / "io" / "image.mha"
    numpy_array64 = numpy_array.astype("int64")

    # Writing int64 data leads to a ValueError by default
    with pytest.raises(ValueError):
        tiger.io.write_image(imagefile, numpy_array64, header)

    tiger.io.write_image(imagefile, numpy_array64, header, allow_int64=True)
    arr, h = tiger.io.read_image(imagefile)
    assert np.array_equal(arr, numpy_array64)


@pytest.mark.parametrize("xy", [(15, 16), (16, 15), (15, 15)])
def test_attempt_write_tif_incompatible_dimensions(xy, tmp_path):
    imagefile = tmp_path / "image.tif"

    image = np.zeros(xy + (16,), dtype="int16")
    with pytest.raises(tiger.io.ImageWriterError):
        tiger.io.write_image(imagefile, image)


@pytest.mark.parametrize("filetype", ["nii", "nii.gz"])
@pytest.mark.parametrize("compression", [False, True])
def test_read_nibabel(filetype, compression, numpy_array, header, tmp_path):
    """Reading from the just written file should reproduce content and header"""
    filename = tmp_path / "io" / f"image.{filetype}"

    # Write to file
    tiger.io.write_image(filename, numpy_array, header, use_compression=compression)
    assert filename.exists()

    # Read back
    reader = tiger.io.NiBabelImageReader()
    arr, h = reader(filename)

    # Compare pixel data
    assert np.array_equal(arr, numpy_array)
    assert arr.dtype == numpy_array.dtype

    # Compare header
    assert h == header

    # Cleanup
    shutil.rmtree(filename.parent)


def test_read_tag_file(resources_path):
    ref_image, ref_header = tiger.io.read_image(resources_path / "sliceomatic.mha")
    tag_image, tag_header = tiger.io.read_image(resources_path / "sliceomatic.tag")

    assert np.array_equal(tag_image, ref_image)
    assert tag_header == ref_header


def test_read_dicom(resources_path):
    # Read from directory
    image, header = tiger.io.read_dicom(resources_path / "dicom")
    assert image.shape == (512, 512, 7)

    headers = {
        "patient_id": "17.09.05-08:31:07-STD-1.3.12.2.1107.5.1.4.60204",
        "patient_age": "022Y",
        "patient_sex": "M",
        "study": "1.3.12.2.1107.5.1.4.60204.30000017090505090890800000005",
        "series": "1.3.12.2.1107.5.1.4.60204.30000017090505121140500010286",
        "modality": "CT",
        "image_type": r"ORIGINAL\PRIMARY\AXIAL\CT_SOM5 SPI",
        "body_part": "EXTREMITY",
        "slice_thickness": 4,
        "convolution_kernel": "B60s",
        "contrast_bolus_agent": None,
        "temporal_position_index": None,
    }
    for key, value in headers.items():
        if value is None:
            assert key not in header
        elif isinstance(value, str):
            assert header[key] == value
        else:
            assert header[key] == pytest.approx(value)

    # Read from list of files
    files = (resources_path / "dicom").glob("*.dcm")
    image2, header2 = tiger.io.read_dicom(files)
    assert np.array_equal(image, image2)
    assert header == header2


def test_read_dicom_reader(resources_path):
    reader = tiger.io.DicomReader()
    images = list(reader(resources_path / "dicom"))
    assert len(images) == 1


def test_read_dicom_reader_single_slice(resources_path):
    files = list((resources_path / "dicom").glob("*.dcm"))
    image, header = tiger.io.read_image(files[0])
    assert image.shape == (512, 512, 1)
    assert header["reader"] == "DicomReader"


def test_read_dicom_none(tmp_path):
    with pytest.raises(ValueError):
        tiger.io.read_dicom(tmp_path)

    reader = tiger.io.DicomReader()
    images = list(reader(tmp_path))
    assert len(images) == 0


def test_discover_dicom(resources_path):
    series = tiger.io.discover_dicom_files(resources_path / "dicom")
    assert len(series) == 1
    assert "1.3.12.2.1107.5.1.4.60204.30000017090505121140500010286" in series

    files = list(series.values())[0]
    assert len(files) == 7


def test_read_dicom_itk(resources_path):
    dicom_path = resources_path / "dicom"
    ref_image, ref_header = tiger.io.read_dicom(dicom_path)

    reader = tiger.io.ItkImageReader()
    image, header = reader(dicom_path.glob("*.dcm"))

    assert image == pytest.approx(ref_image)
    assert header == ref_header


def test_read_dicom_zip(resources_path, tmp_path):
    # Create zip file with DICOM files
    dcmdir = resources_path / "dicom"
    filename = tmp_path / "dicom.zip"
    with zipfile.ZipFile(filename, "w") as zip:
        for dcmfile in dcmdir.glob("*.dcm"):
            zip.write(dcmfile, arcname=dcmfile.name)

    # Read image from zip file
    ref_image, ref_header = tiger.io.read_dicom(dcmdir)
    image, header = tiger.io.read_dicom(filename)

    assert np.array_equal(image, ref_image)
    assert header == ref_header
