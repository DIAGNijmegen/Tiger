Tiger for Medical Image Analysis
================================

.. toctree::
    :hidden:
    :caption: User Guide

    Tutorials <https://github.com/DIAGNijmegen/msk-tiger/tree/stable/tutorials>
    contributing

.. toctree::
    :hidden:
    :includehidden:
    :caption: API Reference

    modules/augmentation
    modules/cluster
    modules/gc
    modules/io
    modules/masks
    modules/meshes
    modules/metrics
    modules/patches
    modules/plots
    modules/random
    modules/resampling
    modules/torch
    modules/screenshots
    modules/utils
    modules/visdom
    modules/workflow

.. toctree::
    :hidden:
    :caption: History

    Releases <https://github.com/DIAGNijmegen/msk-tiger/releases>
    changelog


Tiger is a small collection of utilities for medical image analysis projects in Python 3.

Installation
------------

To install the latest version, open a command prompt an run:

.. prompt:: bash $

   pip install git+https://github.com/DIAGNijmegen/msk-tiger.git@stable

Or install a specific version:

.. prompt:: bash $

   pip install git+https://github.com/DIAGNijmegen/msk-tiger.git@4.0.0

.. note::

    In the official DIAG docker base image, Tiger is already pre-installed.

Getting started
---------------

Most functions in Tiger work with numpy arrays. Because numpy arrays contain only the pixel values, but no additional
information about the image volume such as the spacing between pixels, or the orientation of the image, a central concept
in Tiger is the :class:`tiger.io.ImageMetadata` object. This object represents the header of an image volume and often needs to
be passed to functions in Tiger together with the numpy array.

This is how you read an image into memory:

.. code-block::

    from tiger.io import read_image, write_image

    image, header = read_image("/path/to/image.mha")
    voxel_spacing = header["spacing"]
    write_image("/new/location.nii.gz", image, header)

In this example, ``image`` is a numpy array and ``header`` the corresponding metadata object (which can also be manipulated).

Pathlib is generally supported throughout the library:

.. code-block::

    from pathlib import Path
    from tiger.io import read_image

    image_file = Path("/path/to/image.mha")
    image, header = read_image(image_file)

Modules
-------

+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.augmentation`   | Data augmentation methods for images and segmentation masks.         |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.cluster`        | Utilities for running jobs on SOL, such as dynamic entrypoints.      |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.gc`             | Helper functions for working with the grand-challenge.org API        |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.io`             | Functions for reading and writing images and other data.             |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.masks`          | Finding bounding boxes or connected components in label volumes.     |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.meshes`         | Operations on surface and volume meshes                              |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.metrics`        | Compute volume overlap, surface distances, calcium scores, etc.      |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.patches`        | Helpers for extracting patches from images or masks.                 |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.plots`          | Generate various types of plots in a standardized way.               |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.random`         | Shortcuts for making random decisions with certain probability etc.  |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.resampling`     | Functions for resampling images to different resolution/orientation. |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.screenshots`    | Generate 2D screenshots from 3D image volumes, with overlay.         |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.torch`          | Utilities for PyTorch-based projects, such as additional layers.     |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.utils`          | Other utilities that do not fit into any of the other modules.       |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.visdom`         | Helpers for monitoring experiments with Visdom.                      |
+-----------------------------+----------------------------------------------------------------------+
| :mod:`tiger.workflow`       | Utilities for workflow managment, like keeping track of experiments. |
+-----------------------------+----------------------------------------------------------------------+

Error classes
-------------

.. autoclass:: tiger.TigerException
    :members:

Dependencies
------------

* Python 3.8+
* numpy
* scipy
* scikit-image
* SimpleITK
* ITK
* VTK
* tifffile
* pillow
* matplotlib
* panimg
* gcapi
* python-dateutil
* NiBabel

Some modules require additional packages to be installed, e.g., the :mod:`tiger.torch` module requires PyTorch, or the
:mod:`tiger.visdom` module requires a specific version of Visdom. However, these are not installed by default when installing
Tiger with pip because they are large dependencies or rarely used modules. Not installing PyTorch enables users of Tensorflow
to use Tiger without forcing them to install unnecessary libraries.

Authors
-------

* Nikolas Lessmann
* Michel Kok
* Thijs van den Hout
