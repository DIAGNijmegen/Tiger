import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def random_state():
    return np.random.RandomState(seed=82345)


# Location of resource files
@pytest.fixture(scope="session")
def resources_path():
    return Path(__file__).parent.absolute() / "resources"


# Cache images/masks per session
def read_resource_image(resources_path, filename):
    from tiger.io import read_image

    return read_image(resources_path / filename)


@pytest.fixture(scope="session")
def mri_image(resources_path):
    return read_resource_image(resources_path, "mri_image.mha")


@pytest.fixture(scope="session")
def mri_image_iso(resources_path):
    return read_resource_image(resources_path, "mri_image_iso.mha")


@pytest.fixture(scope="session")
def mri_mask(resources_path):
    return read_resource_image(resources_path, "mri_mask.mha")


@pytest.fixture(scope="session")
def mri_mask_iso(resources_path):
    return read_resource_image(resources_path, "mri_mask_iso.mha")


@pytest.fixture(scope="session")
def ct_image(resources_path):
    return read_resource_image(resources_path, "ct_image.mha")


@pytest.fixture(scope="session")
def ct_image_iso(resources_path):
    return read_resource_image(resources_path, "ct_image_iso_sitk.mha")


@pytest.fixture(scope="session")
def ct_mask(resources_path):
    return read_resource_image(resources_path, "ct_mask.mha")


@pytest.fixture(scope="session")
def ct_mask_iso(resources_path):
    return read_resource_image(resources_path, "ct_mask_iso_sitk.mha")


@pytest.fixture(scope="session")
def ct_mask_iso_dt(resources_path):
    return read_resource_image(resources_path, "ct_mask_iso_dt.mha")


# A codebase for testing workflow/experiment settings classes
@pytest.fixture
def dummy_codebase(tmp_path):
    # Prepare codebase with some dummy code
    codebase = tmp_path / "code"

    codebase.mkdir()
    with open(str(codebase / "Entrypoint.py"), "w") as fp:
        fp.write("from tiger.cluster import Entrypoint\n")
        fp.write("Entrypoint().execute()")
    with open(str(codebase / "PrintArgvFile.py"), "w") as fp:
        fp.write("import sys\n")
        fp.write("print(sys.argv)\n")
        fp.write("print(__file__)")

    # Overwrite sys.argv
    sys_argv = sys.argv.copy()
    sys.argv = [str((codebase / "PrintArgvFile.py").absolute())]

    sys_path = sys.path.copy()
    cwd = os.getcwd()

    yield codebase

    # Reset everything
    shutil.rmtree(codebase)
    sys.argv = sys_argv
    sys.path = sys_path
    os.chdir(cwd)
