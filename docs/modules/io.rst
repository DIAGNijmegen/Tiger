:mod:`tiger.io`
===============

.. automodule:: tiger.io

Image metadata
--------------

.. autoclass:: tiger.io.ImageMetadata
    :members:
    :special-members:

Image reading
-------------

.. warning::

    The pixel data is always returned in x-y-z orientation, i.e., the first axis corresponds to the x-axis, the second to the y-axis, etc.
    This is different from the typical conversion between SimpleITK and numpy!

.. autofunction:: tiger.io.read_image
.. autofunction:: tiger.io.read_dicom

.. autoclass:: ItkImageReader
    :members:

.. autoclass:: NiBabelImageReader
    :members:

.. autoclass:: TiffImageReader
    :members:

.. autoclass:: TagImageReader
    :members:

.. autoclass:: DicomReader
    :members:

.. autoclass:: DiagDicomReader
    :members:

.. autofunction:: tiger.io.discover_dicom_files
.. autofunction:: tiger.io.sort_dicom_files

.. autoexception:: tiger.io.ImageReaderError
    :members:

.. autoexception:: tiger.io.ImageReaderNotAvailableError
    :members:

Image writing
-------------

.. warning::

    Image writers expect the pixel data in x-y-z orientation, i.e., the first axis must correspond to the x-axis, the second to the y-axis,
    etc. This is different from the orientation that SimpleITK expects when converting a numpy array into a SimpleITK image. However, see
    also the note above - the image readers return the pixel data in x-y-z orientation as well.

.. autofunction:: tiger.io.write_image

.. autoclass:: ItkImageWriter
    :members:

.. autoclass:: TiffImageWriter
    :members:

.. autoexception:: tiger.io.ImageWriterError
    :members:

Image conversion
----------------

.. autofunction:: tiger.io.convert_image

.. autofunction:: tiger.io.image_to_sitk
.. autofunction:: tiger.io.image_from_sitk

JSON I/O
--------

.. autofunction:: tiger.io.read_json
.. autofunction:: tiger.io.write_json

Utilities
---------

.. autofunction:: tiger.io.path_exists
.. autofunction:: tiger.io.refresh_file_list
.. autofunction:: tiger.io.checksum
.. autofunction:: tiger.io.copyfile
.. autofunction:: tiger.io.copytree
