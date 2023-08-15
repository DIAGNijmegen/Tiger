Changelog
=========

Release 6.3
-----------

* Add support for numpy data types to :func:`tiger.io.write_json`

Release 6.2
-----------

* Added :func:`tiger.torch.collate_patches`

Release 6.1
-----------

* Added :class:`tiger.io.NiBabelImageReader` as alternative method for reading Nifti images.
* Made smoothing of the mesh an option that is disabled by default in :func:`tiger.meshes.resample_mask_via_mesh`
* Fixed bug in :class:`tiger.plots.LearningCurve` that would lead to nothing being plotted.

Release 6.0
-----------

* Added the new :mod:`tiger.gc` module for working with the grand-challenge.org API.

-----

Release 5.0
-----------

* Added the new :mod:`tiger.meshes` module, which also introduces a dependency on ITK and VTK.

-----

Release 4.9
-----------

* Added :func:`tiger.masks.most_common_labels` that lists all labels sorted by frequency

Release 4.8
-----------

* See GitHub

Release 4.7
-----------

* Fixed bug in :func:`tiger.resampling.reorient_image` that would result in correct images but suboptimal spacing of the new image. In this
  version, the parameter keep_spacing has been replaced with a new_spacing parameter, which allows the user to directly specify a custom
  spacing for the resampled image. The default behavior has not changed.
* Fixed another bug in :func:`tiger.resampling.reorient_image` that would result in incorrect shape calculations, resulting in a tool small
  image.
* Added a new_shape parameter to :func:`tiger.resampling.reorient_image` to enforce a specific target shape.
* Fixed a bug in : func:`tiger.resampling.align_mask_with_image` (wrong keyword argument).
* Added :func:`tiger.utils.count` to count the size of True elements in an iterable.

Release 4.6
-----------

* Added color LUT :class:`tiger.screenshots.SpineColors`
* Added support for captions to :class:`tiger.screenshots.ScreenshotGenerator`
* Several methods and functions in the screenshots module now require keyword arguments.

Release 4.5
-----------

* Fixed bug in :class:`tiger.cluster.Entrypoint` and :class:`tiger.workflow.ExperimentSettings` that would prevent external entrypoints in
  preserved codebases from executing properly.

Release 4.4
-----------

* Added helper function :func:`tiger.io.convert_image` for reading and immediately writing an image file, typically using different formats.

Release 4.3
-----------

* Added option to :func:`tiger.plots.LearningCurve` to plot only some metrics.
* Fixed bug in :func:`tiger.io.ImageMetadata.from_dict` function.
* Removed :mod:`tiger.grandchallenge` module, the components for interacting with the grand-challenge.org API need to be rewritten.
* Added :class:`tiger.masks.ConnectedComponents` that simplifies working with connected components in masks.

Release 4.2
-----------

* Add maximum intensity projection option for the overlay to screenshot generator.
* Replaced :func:`tiger.resampling.change_direction` with :func:`tiger.resampling.reorient_image`
* :func:`tiger.resampling.align_images` no longer requires a reference image but only its shape.

Release 4.1
-----------

* The function :func:`tiger.masks.merge_masks` no longer returns the header of the merged masks since this is identical to the
  reference header that is an input of the function.
* Added the temporal position index to :class:`tiger.io.ImageMetadata`.
* Added a helper for making screenshots in interesting places, the :func:`tiger.screenshots.find_center` function.
* Added :func:`tiger.utils.first` to return the first item in an iterable object.

Release 4.0
-----------

* The minimal Python version is now 3.8
* The :class:`tiger.io.DicomReader` now uses the panimg package to read DICOM files. The interface has changed, the reader now
  supports reading multiple images at once in case multiple series are found.
* :func:`tiger.io.discover_dicom_files` does now return all series and not just the files that belong to a single series.

-----

Release 3.10
-----------

* The :class:`tiger.io.ItkImageReader` now sorts DICOM files if a list of filenames is provided.

Release 3.9
-----------

* :class:`tiger.io.TagImageReader` now returns a mutable numpy array.
* Added :mod:`tiger.grandchallenge` to download and parse answers from Grand Challenge Reader Studies.
* Added :func:`tiger.resampling.align_images` and :func:`tiger.resampling.align_mask_with_image`
* Image headers now have a method `has_same_world_matrix(other)` to compare only the coordinate system of two images.

Release 3.8
-----------

* Added a dtype parameter to :func:`tiger.masks.merge_masks` since previously the merged mask had always "double" as data type.
* :class:`tiger.screenshots.ScreenshotCollage` now supports resampling through the screenshot generators.
* Added :class:`tiger.io.TagImageReader` to read the sliceOmatic output format.
* Fixed a bug in :class:`tiger.plots.LearningCurve` that would prevent creating instances of the class.

Release 3.7
-----------

* :func:`tiger.io.read_dicom` and the :class:`tiger.io.DicomReader` are now able to discover DICOM files in a folder and sort them
  correctly. For this purpose, :func:`tiger.io.discover_dicom_files` was added.
* Added an auto_plot mode to :class:`tiger.plots.LearningCurve`.
* Fixed bugs in sliding window patch extractors and reworked their interface.
* Added support for a Gaussian weight map to the sliding window iterators.
* Renamed parameters of Dice score loss functions to match the naming convention in PyTorch.
* Replaced function for computation of Bland-Altman limits of agreement with a class.

Release 3.6
-----------

* Corrected return type of :func:`tiger.utils.slice_nd_array`.
* :func:`tiger.io.read_json` now returns OrderedDict instead of dict by default.
* Added :class:`tiger.plots.LearningCurve` for saving loss values and plotting learning curves.
* Added :class:`tiger.torch.PolynomialLearningRateDecay` (a learning rate scheduler).
* Fixed a bug in :func:`tiger.io.path_exists` that would make it crash if also the parent directory does not exist.

Release 3.5
-----------

* Removed :class:`tiger.torch.Flatten` since :class:`torch.nn.Flatten` is now available and serves the same purpose.
* :class:`tiger.cluster.EntrypointError` is now based on :class:`tiger.TigerException`.
* Fixed an issue where :class:`tiger.io.DiagDicomReader` would fail in jupyter notebooks.
* Added a `has_direction()` function to :class:`tiger.io.ImageMetadata` that checks if the image has a specific orientation.
* :class:`tiger.screenshots.ScreenshotGenerator` supports now resampling to a standard resolution.
* Added the sliding window patch extractors :class:`tiger.patches.SlidingRect` and :class:`tiger.patches.SlidingCuboid`.

Release 3.4
-----------

* New functions :func:`tiger.io.path_exists` and :func:`tiger.io.refresh_file_list` for working with files on network shares.
* New function :func:`tiger.io.checksum` for computing hash-based checksums for files.
* New function :func:`tiger.screenshots.imshow` that helps showing slices from 3D images in jupyter notebooks etc.
* Updated :func:`tiger.metrics.mean_dice_score` to require a list of labels instead of getting the labels from one of the masks.
* Removed `threshold` parameter from :func:`tiger.resampling.resample_mask_dt` function, threshold is now automatically computed.

.. note::

    The changelog was introduced with Tiger 3.4, earlier changes are documented only in the git commit history.
