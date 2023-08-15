from pathlib import Path

import numpy as np
import pytest

from tiger.io import ImageMetadata, read_image, write_image
from tiger.meshes import resample_mask_via_mesh


@pytest.mark.parametrize("label", [1, 0])
def test_resample_mask_via_mesh(label, tmpdir):
    # Create a dummy mask
    mask = np.zeros((40, 40, 40), dtype="uint8")
    mask[15:25, 15:25, 15:25] = label
    mask_header = ImageMetadata(ndim=3, spacing=(2, 2, 2))
    mask_file = Path(tmpdir) / "mask_in.mha"
    write_image(mask_file, mask, mask_header)

    # Create a dummy image with different resolution
    image = np.zeros((80, 80, 80), dtype="int16")
    header = ImageMetadata(ndim=3, spacing=(1, 1, 1))
    img_file = Path(tmpdir) / "image.mha"
    write_image(img_file, image, header)

    # Resample mask to image resolution using a surface mesh
    dst_file = Path(tmpdir) / "mask_out.mha"
    obj_file = Path(tmpdir) / "mesh.obj"
    resample_mask_via_mesh(
        src_file=mask_file, dst_file=dst_file, reference_image=img_file, mesh_file=obj_file
    )

    assert dst_file.exists()

    if label == 1:
        assert obj_file.exists()
    else:
        # Empty mask will result in no mesh
        assert not obj_file.exists()

    new_mask, new_header = read_image(dst_file)
    assert new_mask.shape == image.shape
    assert set(np.unique(new_mask)) == set(np.unique(mask))
    assert new_header.has_same_world_matrix(header)
