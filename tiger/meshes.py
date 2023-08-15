from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Set, Union

import itk
import numpy as np
import SimpleITK as sitk
import vtk

from .io import PathLike


def resample_mask_via_mesh(
    *,
    src_file: PathLike,
    dst_file: PathLike,
    reference_image: PathLike,
    mesh_file: Optional[PathLike] = None,
    smoothing: Union[bool, Set[int]] = False,
):
    """Converts a mask into a surface mesh and back into a mask at a different resolution (and orientation if needed)"""
    dst_file = Path(dst_file)
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    # Read target shape and coordinate system from reference image file
    ref = sitk.ImageFileReader()
    ref.SetFileName(str(reference_image))
    ref.ReadImageInformation()

    # Read the mask itself into memory and obtain a list of labels
    mask = itk.imread(str(src_file))
    labels = [int(v) for v in sorted(np.unique(itk.GetArrayViewFromImage(mask))) if v > 0]

    if len(labels) == 0:
        empty_mask = sitk.Image(ref.GetSize(), ref.GetPixelIDValue())
        sitk.WriteImage(empty_mask, str(dst_file), useCompression=True)
        return

    # Determine which labels to smooth
    if not smoothing:
        objects_to_smooth = set()
    else:
        try:
            objects_to_smooth = set(smoothing)
        except TypeError:
            objects_to_smooth = set(labels)

    # Compute dimensions of target mask, including a little margin that we will cut away
    # later (this helps prevent artifacts)
    rotation = np.asarray(ref.GetDirection()).reshape(3, 3)
    rotated_spacing = rotation @ ref.GetSpacing()

    target_size = tuple(s + 2 for s in ref.GetSize())
    target_spacing = ref.GetSpacing()
    target_direction = itk.matrix_from_array(rotation)
    target_origin = tuple(o - s for o, s in zip(ref.GetOrigin(), rotated_spacing))

    combined_mask = None
    combined_mesh = vtk.vtkAppendPolyData()

    with TemporaryDirectory() as tmpdir:
        for label in labels:
            tmp_mesh_file = Path(tmpdir) / f"mesh{label}.vtk"

            # Convert one structure into a mesh
            mesh = itk.binary_mask3_d_mesh_source(mask, object_value=label)

            # Transfer via file from ITK to VTK
            itk.meshwrite(mesh, str(tmp_mesh_file))
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(tmp_mesh_file))
            reader.Update()

            # Smooth mesh
            if label in objects_to_smooth:
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputConnection(reader.GetOutputPort())
                smoother.SetNumberOfIterations(20)
                smoother.BoundarySmoothingOn()
                smoother.FeatureEdgeSmoothingOn()
                smoother.SetFeatureAngle(120)
                smoother.SetPassBand(0.001)
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn()
                smoother.Update()
            else:
                smoother = reader

            # Add mesh of this structure to the big mesh with all structures
            combined_mesh.AddInputConnection(smoother.GetOutputPort())
            combined_mesh.Update()

            # Transfer back to ITK via a file
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileVersion(42)
            writer.SetInputConnection(smoother.GetOutputPort())
            writer.SetFileName(str(tmp_mesh_file))
            writer.Write()
            mesh = itk.meshread(str(tmp_mesh_file))

            # Convert mesh into mask at the original resolution
            new_mask = itk.triangle_mesh_to_binary_image_filter(
                mesh,
                inside_value=label,
                outside_value=0,
                size=target_size,
                spacing=target_spacing,
                direction=target_direction,
                origin=target_origin,
            )

            if combined_mask is None:
                combined_mask = new_mask.astype(itk.SS)
            else:
                combined_array = itk.GetArrayViewFromImage(combined_mask)
                indices = combined_array == 0
                combined_array[indices] = itk.GetArrayViewFromImage(new_mask)[indices]

    # Cut away margin
    cropped_mask = itk.crop_image_filter(combined_mask, boundary_crop_size=(1, 1, 1))
    itk.imwrite(cropped_mask, filename=str(dst_file), compression=True)

    # Store mesh as well?
    if mesh_file is not None:
        mesh_file = Path(mesh_file)
        mesh_file.parent.mkdir(parents=True, exist_ok=True)

        writer = vtk.vtkOBJWriter()
        writer.SetInputConnection(combined_mesh.GetOutputPort())
        writer.SetFileName(str(mesh_file))
        writer.Write()
