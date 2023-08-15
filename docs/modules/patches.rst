:mod:`tiger.patches`
====================

.. automodule:: tiger.patches

Patch extraction
----------------

.. autoclass:: tiger.patches.PatchExtractor2D
    :members:

.. autoclass:: tiger.patches.PatchExtractor3D
    :members:

Sliding window iterators
------------------------

.. autoclass:: tiger.patches.SlidingRect
    :members:
    :inherited-members:

.. autoclass:: tiger.patches.SlidingCuboid
    :members:
    :inherited-members:

.. autoclass:: tiger.patches.Patch
    :members:

Utils
-----

.. autofunction:: tiger.patches.compute_crop
.. autofunction:: tiger.patches.compute_valid_patch_pairs
.. autofunction:: tiger.patches.extract_valid_patch_pairs
