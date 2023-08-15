:mod:`tiger.resampling`
=======================

.. automodule:: tiger.resampling

.. autofunction:: tiger.resampling.resample_image
.. autofunction:: tiger.resampling.resample_image_itk

.. autofunction:: tiger.resampling.resample_mask
.. autofunction:: tiger.resampling.resample_mask_itk
.. autofunction:: tiger.resampling.resample_mask_dt

Coordinate system orientation
-----------------------------

.. autofunction:: tiger.resampling.reorient_image
.. autofunction:: tiger.resampling.normalize_direction_simple
.. autofunction:: tiger.resampling.restore_original_direction_simple

.. autofunction:: tiger.resampling.align_images
.. autofunction:: tiger.resampling.align_mask_with_image

Non-standard resamplers
-----------------------

.. autoclass:: tiger.resampling.WeightedAverageResampler
    :members:

Utilities
---------

.. autofunction:: tiger.resampling.pad_or_crop_image
