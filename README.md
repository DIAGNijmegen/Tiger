[![Tests](https://github.com/DIAGNijmegen/msk-tiger/workflows/Tests/badge.svg)](https://github.com/DIAGNijmegen/msk-tiger/actions)
[![Build](https://github.com/DIAGNijmegen/msk-tiger/workflows/Build/badge.svg)](https://github.com/DIAGNijmegen/msk-tiger/releases)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Tiger

Tiger is a collection of utilities for medical image analysis projects in Python 3.

To install the latest version, open a command line an run:

```shell script
pip install git+https://github.com/DIAGNijmegen/Tiger.git@stable
```

In the official DIAG docker base image, Tiger is already pre-installed.

## Getting started

Most functions in Tiger work with numpy arrays. Because numpy arrays contain only the pixel values mask, but no additional
information about the image volume such as the spacing between pixels, or the orientation of the image, a central concept
in Tiger is the `tiger.io.ImageMetadata` object. This object represents the header of an image volume and often needs to
be passed to functions in Tiger together with the numpy array.

This is how you read an image into memory:

```python
from tiger.io import read_image, write_image

image, header = read_image("/path/to/image.mha")
voxel_spacing = header["spacing"]
write_image("/new/location.nii.gz", image, header)
```

In this example, `image` is a numpy array and `header` the corresponding metadata object (which can also be manipulated).

Pathlib is generally supported throughout the library.

## Contributing

Please report bugs and feature requests here on GitHub. Contributions in the form of pull requests are very welcome.
