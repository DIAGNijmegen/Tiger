from pathlib import Path

import pytest

import tiger.io


def test_refresh_file_list(tmp_path: Path):
    tiger.io.refresh_file_list(tmp_path)

    with pytest.raises(IOError):
        tiger.io.refresh_file_list(tmp_path / "foo.txt")


def test_exists(tmp_path: Path):
    assert tiger.io.path_exists(tmp_path)

    # File within temporary directory
    tmpfile = tmp_path / "foo.txt"
    assert not tiger.io.path_exists(tmpfile)
    with tmpfile.open("w") as fp:
        fp.write("Hello world")
    assert tiger.io.path_exists(tmpfile)

    # Directory within temporary directory
    tmp_path_dir = tmp_path / "bar"
    assert not tiger.io.path_exists(tmp_path_dir)
    tmp_path_dir.mkdir()
    assert tiger.io.path_exists(tmp_path_dir)


def test_checksum(tmp_path: Path):
    tmpfile = tmp_path / "foo.txt"

    # Should raise an exception if the file does not exist
    assert not tmpfile.exists()
    with pytest.raises(FileNotFoundError):
        tiger.io.checksum(tmpfile)

    # Create dummy file with known hash
    expected_hash = "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c"
    with tmpfile.open("w") as fp:
        fp.write("Hello world")
    assert tiger.io.checksum(tmpfile) == expected_hash

    # Use a string instead of a Path object
    assert tiger.io.checksum(str(tmpfile)) == expected_hash

    # Use a different block size
    assert tiger.io.checksum(tmpfile, chunk_size=2048) == expected_hash

    # Use a different hashing algorithm
    assert tiger.io.checksum(tmpfile, algorithm="md5") == "3e25960a79dbc69b674cd4ec67a72c62"

    # Try to use an unknown hashing algorithm
    with pytest.raises(ValueError):
        tiger.io.checksum(tmpfile, algorithm="md500")

    # Try to compute checksum for a directory
    with pytest.raises(ValueError):
        tiger.io.checksum(tmp_path)

    # Change file, check again
    with tmpfile.open("a") as fp:
        fp.write("!")
    assert tiger.io.checksum(tmpfile) != expected_hash
