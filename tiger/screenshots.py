import colorsys
import itertools
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from scipy import ndimage

from .resampling import resample_image, resample_mask

# Colors are defined in the RGBA system with values from 0 to 255
Color = Tuple[int, int, int, int]

# Color look up tables return colors for labels
LUT = Callable[[float], Color]


def change_opacity(color: Color, opacity: float) -> Color:
    return color[:3] + (int(round(255 * opacity)),)


class DefaultColors:
    """Maps neighboring labels to distinct colors (derived from Mevislab's LUT500Colors module)"""

    def __init__(self):
        self.colors: Sequence[Color] = (  # RGBA
            (0, 0, 0, 0),
            (255, 0, 0, 255),
            (224, 255, 0, 255),
            (0, 255, 59, 255),
            (0, 166, 255, 255),
            (120, 0, 255, 255),
            (255, 0, 107, 255),
            (181, 127, 0, 255),
            (33, 181, 0, 255),
            (0, 181, 168, 255),
            (9, 0, 181, 255),
            (181, 0, 150, 255),
            (128, 36, 0, 255),
            (76, 128, 0, 255),
            (0, 128, 66, 255),
            (0, 47, 128, 255),
            (96, 0, 128, 255),
            (128, 0, 17, 255),
            (255, 252, 87, 255),
            (87, 255, 105, 255),
            (87, 217, 255, 255),
            (143, 87, 255, 255),
            (255, 87, 178, 255),
            (181, 130, 62, 255),
            (100, 181, 62, 255),
            (62, 181, 158, 255),
            (62, 71, 181, 255),
            (181, 62, 177, 255),
            (128, 56, 43, 255),
            (105, 128, 43, 255),
            (43, 128, 75, 255),
            (43, 85, 128, 255),
            (96, 43, 128, 255),
            (128, 43, 66, 255),
            (255, 240, 145, 255),
            (148, 255, 145, 255),
            (145, 247, 255, 255),
            (168, 145, 255, 255),
            (255, 145, 219, 255),
            (181, 136, 103, 255),
            (138, 181, 103, 255),
            (103, 181, 156, 255),
            (103, 119, 181, 255),
            (174, 103, 181, 255),
            (128, 74, 73, 255),
            (120, 128, 73, 255),
            (73, 128, 87, 255),
            (73, 107, 128, 255),
            (99, 73, 128, 255),
            (128, 73, 94, 255),
            (255, 184, 0, 255),
            (43, 255, 0, 255),
            (0, 255, 242, 255),
            (18, 0, 255, 255),
            (255, 0, 209, 255),
            (181, 54, 0, 255),
            (105, 181, 0, 255),
            (0, 181, 96, 255),
            (0, 63, 181, 255),
            (139, 0, 181, 255),
            (181, 0, 22, 255),
            (128, 128, 0, 255),
            (0, 128, 15, 255),
            (0, 98, 128, 255),
            (45, 0, 128, 255),
            (128, 0, 68, 255),
            (255, 184, 87, 255),
            (138, 255, 87, 255),
            (87, 255, 224, 255),
            (87, 99, 255, 255),
            (255, 87, 247, 255),
            (181, 81, 62, 255),
            (148, 181, 62, 255),
            (62, 181, 109, 255),
            (62, 119, 181, 255),
            (138, 62, 181, 255),
            (181, 62, 92, 255),
            (128, 116, 43, 255),
            (45, 128, 43, 255),
            (43, 119, 128, 255),
            (62, 43, 128, 255),
            (128, 43, 99, 255),
            (255, 194, 145, 255),
            (191, 255, 145, 255),
            (145, 255, 219, 255),
            (145, 168, 255, 255),
            (247, 145, 255, 255),
            (181, 105, 103, 255),
            (170, 181, 103, 255),
            (103, 181, 125, 255),
            (103, 152, 181, 255),
            (141, 103, 181, 255),
            (181, 103, 132, 255),
            (128, 112, 73, 255),
            (80, 128, 73, 255),
            (73, 128, 126, 255),
            (76, 73, 128, 255),
            (128, 73, 116, 255),
            (255, 82, 0, 255),
            (145, 255, 0, 255),
            (0, 255, 140, 255),
            (0, 84, 255, 255),
            (199, 0, 255, 255),
            (255, 0, 26, 255),
            (177, 181, 0, 255),
            (0, 181, 24, 255),
            (0, 136, 181, 255),
            (67, 0, 181, 255),
            (181, 0, 94, 255),
            (128, 76, 0, 255),
            (36, 128, 0, 255),
            (0, 128, 106, 255),
            (0, 6, 128, 255),
            (128, 0, 119, 255),
            (255, 117, 87, 255),
            (204, 255, 87, 255),
            (87, 255, 158, 255),
            (87, 166, 255, 255),
            (196, 87, 255, 255),
            (255, 87, 125, 255),
            (181, 167, 62, 255),
            (62, 181, 62, 255),
            (62, 167, 181, 255),
            (89, 62, 181, 255),
            (181, 62, 139, 255),
            (128, 83, 43, 255),
            (79, 128, 43, 255),
            (43, 128, 102, 255),
            (43, 59, 128, 255),
            (121, 43, 128, 255),
            (255, 150, 145, 255),
            (237, 255, 145, 255),
            (145, 255, 176, 255),
            (145, 212, 255, 255),
            (201, 145, 255, 255),
            (255, 145, 186, 255),
            (181, 161, 103, 255),
            (114, 181, 103, 255),
            (103, 181, 179, 255),
            (110, 103, 181, 255),
            (181, 103, 165, 255),
            (128, 91, 73, 255),
            (103, 128, 73, 255),
            (73, 128, 105, 255),
            (73, 91, 128, 255),
            (116, 73, 128, 255),
            (128, 73, 76, 255),
            (247, 255, 0, 255),
            (0, 255, 38, 255),
            (0, 186, 255, 255),
            (97, 0, 255, 255),
            (255, 0, 128, 255),
            (181, 112, 0, 255),
            (49, 181, 0, 255),
            (0, 181, 154, 255),
            (0, 5, 181, 255),
            (181, 0, 167, 255),
            (128, 26, 0, 255),
            (87, 128, 0, 255),
            (0, 128, 55, 255),
            (0, 57, 128, 255),
            (85, 0, 128, 255),
            (128, 0, 28, 255),
            (255, 237, 87, 255),
            (87, 255, 89, 255),
            (87, 232, 255, 255),
            (130, 87, 255, 255),
            (255, 87, 194, 255),
            (181, 119, 62, 255),
            (109, 181, 62, 255),
            (62, 181, 148, 255),
            (62, 81, 181, 255),
            (176, 62, 181, 255),
            (128, 50, 43, 255),
            (112, 128, 43, 255),
            (43, 128, 69, 255),
            (43, 92, 128, 255),
            (88, 43, 128, 255),
            (128, 43, 73, 255),
            (255, 230, 145, 255),
            (158, 255, 145, 255),
            (145, 255, 255, 255),
            (158, 145, 255, 255),
            (255, 145, 230, 255),
            (181, 130, 103, 255),
            (145, 181, 103, 255),
            (103, 181, 148, 255),
            (103, 127, 181, 255),
            (167, 103, 181, 255),
            (181, 103, 109, 255),
            (125, 128, 73, 255),
            (73, 128, 82, 255),
            (73, 112, 128, 255),
            (94, 73, 128, 255),
            (128, 73, 99, 255),
            (255, 161, 0, 255),
            (64, 255, 0, 255),
            (0, 255, 222, 255),
            (0, 5, 255, 255),
            (255, 0, 230, 255),
            (181, 40, 0, 255),
            (121, 181, 0, 255),
            (0, 181, 81, 255),
            (0, 78, 181, 255),
            (123, 0, 181, 255),
            (181, 0, 36, 255),
            (128, 117, 0, 255),
            (0, 128, 4, 255),
            (0, 108, 128, 255),
            (34, 0, 128, 255),
            (128, 0, 79, 255),
            (255, 171, 87, 255),
            (150, 255, 87, 255),
            (87, 255, 209, 255),
            (87, 112, 255, 255),
            (250, 87, 255, 255),
            (181, 71, 62, 255),
            (158, 181, 62, 255),
            (62, 181, 100, 255),
            (62, 130, 181, 255),
            (127, 62, 181, 255),
            (181, 62, 101, 255),
            (128, 108, 43, 255),
            (52, 128, 43, 255),
            (43, 126, 128, 255),
            (55, 43, 128, 255),
            (128, 43, 106, 255),
            (255, 186, 145, 255),
            (201, 255, 145, 255),
            (145, 255, 212, 255),
            (145, 176, 255, 255),
            (237, 145, 255, 255),
            (255, 145, 150, 255),
            (176, 181, 103, 255),
            (103, 181, 118, 255),
            (103, 158, 181, 255),
            (136, 103, 181, 255),
            (181, 103, 139, 255),
            (128, 108, 73, 255),
            (85, 128, 73, 255),
            (73, 128, 121, 255),
            (73, 73, 128, 255),
            (128, 73, 121, 255),
            (255, 59, 0, 255),
            (166, 255, 0, 255),
            (0, 255, 120, 255),
            (0, 107, 255, 255),
            (178, 0, 255, 255),
            (255, 0, 46, 255),
            (181, 168, 0, 255),
            (0, 181, 9, 255),
            (0, 150, 181, 255),
            (51, 0, 181, 255),
            (181, 0, 109, 255),
            (128, 66, 0, 255),
            (47, 128, 0, 255),
        )

    def __call__(self, label: float) -> Color:
        """Returns an RGBA color for an integer label (non-integer labels are rounded)"""
        return self.colors[int(np.round(label)) % len(self.colors)]

    def __len__(self) -> int:
        """Number of distinct colors"""
        return len(self.colors)

    def __iter__(self) -> Iterator[Color]:
        yield from self.colors


class VertebraColors(DefaultColors):
    """
    Maps labels from 1-100 to distinct colors and labels 101-200 to the same colors but with
    reduced opacity. This is intended for vertebra segmentations, where label 105 corresponds
    to vertebra 5 but incompletely visible in the image - in an overlay, this vertebra would
    appear in the same color as vertebra with label 5, but the higher transparency indicates
    that the vertebra is not completely visible or was classified as incomplete.
    """

    def __init__(self, incomplete_opacity: float = 0.4):
        super().__init__()

        # First 100 colors copied from default colors,
        # followed by the same 100 colors again, but with decreased opacity
        # (in the pipeline output, labels + 100 are used for incomplete vertebrae)
        self.colors = tuple(
            self.colors[label]
            if label < 100
            else change_opacity(self.colors[label - 100], incomplete_opacity)
            for label in range(200)
        )


class SpineColors(VertebraColors):
    """Adds colors for spinal canal (100) and intervertebral discs (200-300) to the Vertebra color table"""

    def __init__(self, incomplete_opacity: float = 0.4, disc_opacity: float = 0.7):
        super().__init__(incomplete_opacity)
        colors = list(self.colors)

        # Spinal canal color (insert as 100 and 200)
        colors[100] = (255, 185, 21, 255)
        colors.append(change_opacity(colors[100], incomplete_opacity))

        # Intervertebral discs (labels 200-300)
        for i in range(201, 301):
            colors.append(change_opacity(colors[i - 197][:3], disc_opacity))

        self.colors = tuple(colors)


class RibColors:
    def __init__(self):
        self.lut = VertebraColors()

    def __call__(self, label):
        if 100 < label < 114:
            return self.lut(label - 100 + 7)
        elif 200 < label < 214:
            return self.lut(label - 200 + 7)
        elif label == 1:
            return self.lut(1)
        else:
            return self.lut(0)


class TrafficLightColors:
    """Colors from green (1) to red (255)"""

    def __init__(self, saturation: float = 1.0, value: float = 255.0):
        self.colors = [(0, 0, 0, 0)]

        for label in range(1, 256):
            hue = 120 / 360 * (1 - (label - 1) / (255 - 1))
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self.colors.append(rgb + (255,))

    def __call__(self, label):
        return self.colors[label % 256]


class ScreenshotGenerator:
    """
    Creates a 2D screenshot, which can be saved as PNG or similar, from a 3D volume. Optionally,
    an overlay can be added to show segmentation results.

    Note that no resampling is perfomed and that it therefore usually makes sense to first ensure
    that the image and mask have isotropic resolution along the sampling axis.

    Parameters
    ----------
    coordinate
        Relative coordinate along the specified axis, 0 for first slice, 1 for last slice

    axis
        Axis along which the screenshot will be taken from the 3D image - axis=2 corresponds
        to a screenshot along the z axis, i.e., an axial slice.

    window_level
        Optional pair of window and level for normalizing the grayvalues of the image.

    overlay_lut
        A callable object that will be used to translate mask labels to colors when adding an
        overlay to the screenshot. Defaults to :class:`tiger.screenshots.DefaultColors`

    overlay_opacity
        Transparency of the overlay (0 = completely transparent, 1 = completely opaque)

    slab_thickness
        Number of slices along the sampling axis that will be averaged to form the screenshot.

    pixel_spacing
        Desired pixel sizes. Will require the voxel sizes of the image from which a screenshot
        is generated, and will resample the provided image to the desired resolution.

    caption
        Text that is added below the screenshot

    caption_size
        Font size of the caption

    caption_color
        Font color of the caption

    background
        Color of areas that is not occupied by the screenshot or text

    Raises
    ------
    ValueError
        If coordinate, overlay_opacity or slab_thickness are out of range
    """

    def __init__(
        self,
        coordinate: float = 0.5,
        axis: int = 2,
        *,
        window_level: Optional[Tuple[float, float]] = None,
        overlay_lut: Optional[LUT] = None,
        overlay_opacity: float = 0.75,
        overlay_mip: bool = False,
        slab_thickness: int = 1,
        pixel_spacing: Optional[Tuple[float, float]] = None,
        caption: Optional[str] = None,
        caption_size: int = 24,
        caption_color: str = "white",
        background: str = "black",
    ):
        self.coordinate = float(coordinate)
        self.axis = axis
        self.window_level = window_level
        self.overlay_lut = np.vectorize(DefaultColors() if overlay_lut is None else overlay_lut)
        self.overlay_opacity = float(overlay_opacity)
        self.overlay_mip = overlay_mip
        self.slab_thickness = int(slab_thickness)  # in voxels, not mm!
        self.pixel_spacing = pixel_spacing
        self.caption = caption
        self.caption_size = caption_size
        self.caption_color = caption_color
        self.background = background

        if not 0 <= self.coordinate <= 1:
            raise ValueError("Coordinate has to be a value between 0 and 1")
        if not 0 <= self.overlay_opacity <= 1:
            raise ValueError("Overlay opacity has to be a value between 0 and 1")
        if self.slab_thickness < 0 or self.slab_thickness % 2 == 0:
            raise ValueError("Slab thickness has to be an odd number")

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        *,
        spacing: Optional[Iterable[float]] = None,
        caption: Optional[str] = None,
    ) -> PILImage.Image:
        """Turns an image and optionally a corresponding mask into a screenshot (2D image)"""
        image = np.asarray(image)
        has_mask = mask is not None
        resampling = self.pixel_spacing is not None

        if has_mask:
            mask = np.asarray(mask)
            if mask.shape != image.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image shape {image.shape}"
                )

        if resampling:
            if spacing is None:
                raise ValueError(
                    "Pixel spacing of the screenshot is fixed, spacing of the image values is therefore a required parameter"
                )
            else:
                pixel_spacing = list(spacing)
                if len(pixel_spacing) != image.ndim:
                    raise ValueError(
                        "Pixel spacing is required for each individual axis of the image"
                    )

        if image.ndim == 2:
            image = np.expand_dims(image, axis=self.axis)
            if has_mask:
                mask = np.expand_dims(mask, axis=self.axis)
        elif image.ndim == 3:
            if resampling:
                del pixel_spacing[self.axis]
        else:
            raise ValueError(f"Expected 2D or 3D input image, got {image.ndim}D array")

        # Extract 2D slice from 3D volume
        axis_length = image.shape[self.axis]
        max_coordinate = axis_length - 1
        c = np.clip(int(np.round(max_coordinate * self.coordinate)), 0, max_coordinate)

        if self.slab_thickness == 1:
            image_slice = image.take(c, axis=self.axis)
        else:
            slab_offset = (self.slab_thickness - 1) // 2
            indices = [
                i for i in np.arange(c - slab_offset, c + slab_offset + 1) if 0 <= i < axis_length
            ]
            image_slice = image.take(indices, axis=self.axis).mean(axis=self.axis)

        # Resample image?
        if resampling:
            if np.allclose(self.pixel_spacing, pixel_spacing, rtol=0, atol=0.01):
                resampling = False
            else:
                image_slice = resample_image(image_slice, pixel_spacing, self.pixel_spacing)  # type: ignore

        # Scale image values to [0,255]
        if self.window_level is not None:
            window, level = self.window_level
            min_val = level - (window / 2)
            max_val = window
        else:
            min_val = np.min(image_slice.astype(float))
            max_val = np.max(image_slice.astype(float)) - min_val

        image_slice = image_slice.astype(float)
        image_slice -= min_val
        if max_val < 1e-8:
            image_slice.fill(0)
        else:
            image_slice = np.clip(np.round(image_slice / max_val * 255), 0, 255)

        screenshot = PILImage.fromarray(image_slice.astype("uint8"))

        # Turn labels into colors
        if has_mask:
            if self.overlay_mip:
                mask_slice = np.max(mask, axis=self.axis)
            else:
                mask_slice = mask.take(c, axis=self.axis)
            if resampling:
                mask_slice = resample_mask(mask_slice, pixel_spacing, self.pixel_spacing)

            overlay_slice = np.transpose(self.overlay_lut(mask_slice), (1, 2, 0)).astype("uint8")
            overlay = PILImage.fromarray(overlay_slice, mode="RGBA")

            # Place the overlay on top of the image (piece by piece, because blending is only possible
            # for a single alpha value)
            screenshot = screenshot.convert("RGBA")
            for opacity in np.unique(overlay_slice[:, :, 3]):
                if opacity == 0:
                    continue

                # Blend image with overlay
                blended_screenshot = PILImage.blend(
                    screenshot, overlay, alpha=opacity / 255 * self.overlay_opacity
                )

                # Restore parts with different opacity in the overlay, they are blended in a different iteration
                other_opacity = np.zeros_like(overlay_slice, dtype="uint8")
                other_opacity[overlay_slice[:, :, 3] != opacity] = 255
                blended_screenshot.paste(screenshot, None, PILImage.fromarray(other_opacity))

                screenshot = blended_screenshot

        # Correct orientation
        screenshot = screenshot.transpose(PILImage.TRANSPOSE)
        if self.axis != 2:
            screenshot = screenshot.transpose(PILImage.FLIP_TOP_BOTTOM)

        if caption or self.caption:
            if caption is None:
                caption = self.caption

            # Create a new empty image with additional space for the caption
            caption_margin = int(round(self.caption_size * 1.2))
            new_size = (screenshot.size[0], screenshot.size[1] + self.caption_size + caption_margin)
            screenshot_with_caption = PILImage.new("RGBA", new_size, color=self.background)

            # Copy over the screenshot
            screenshot_with_caption.paste(screenshot)

            # Add caption
            font = ImageFont.truetype(
                str(Path(__file__).parent / "resources" / "fonts" / "NotoSans-Regular.ttf"),
                self.caption_size,
            )

            pen = ImageDraw.Draw(screenshot_with_caption)
            pen.text(
                xy=(
                    screenshot.size[0] // 2,
                    screenshot.size[1] + self.caption_size + caption_margin // 3,
                ),
                text=caption,
                font=font,
                fill=self.caption_color,
                anchor="ms",
            )

            # Continue with new image that includes screenshot and caption
            screenshot = screenshot_with_caption

        # Return as RGB image
        return screenshot.convert("RGB")


# Collages are defined using a flexible format in which either a single image/mask can be specified,
# or a 1D vector of images/masks to genereate a collage with one row, or a 2D matrix of images/masks
# to generate a collage with multiple rows (but each row can have a different number of screenshots).
T = TypeVar("T")
CollageGrid = Union[T, Iterable[Union[T, Iterable[T]]]]


class ScreenshotCollage:
    """
    Creates an image with multiple screenshots in a specified layout. Screenshots are generated by
    invoking several screenshot generators.

    Parameters
    ----------
    generators
        A list of generators to be used for creating the screenshot collage. Can either be a single
        generator for a single screenshot, or a list of generators for a row of screenshots next to
        each other, or a list of lists of generators for a grid of screenshots (rows do not need to
        have the same number of screenshots!).

    spacing
        Space in pixels between screenshots

    margin
        Space around the collage to the top/bottom/left/right in pixels

    background
        Color of the background (area occupied by spacing and margin)

    Raises
    ------
    ValueError
        If the generators do not describe a valid grid for a screenshot collage
    """

    def __init__(
        self,
        generators: CollageGrid[ScreenshotGenerator],
        *,
        spacing: int = 1,
        margin: int = 0,
        background: str = "black",
    ):
        try:
            if isinstance(generators, ScreenshotGenerator):
                self.generators: Tuple[Tuple[ScreenshotGenerator, ...], ...] = ((generators,),)
            elif all(isinstance(g, ScreenshotGenerator) for g in generators):
                self.generators = (tuple(generators),)  # type: ignore
            else:
                self.generators = tuple(
                    (g,) if isinstance(g, ScreenshotGenerator) else tuple(g) for g in generators
                )
        except TypeError as e:
            raise ValueError(
                "Generators need to be one or many instances of ScreenshotGenerator"
            ) from e

        self.spacing = int(spacing)
        self.margin = int(margin)
        self.background = background

    def __call__(
        self,
        image: CollageGrid[np.ndarray],
        mask: Optional[CollageGrid[np.ndarray]] = None,
        *,
        spacing: Optional[CollageGrid[Iterable[float]]] = None,
        caption: Optional[CollageGrid[str]] = None,
    ) -> PILImage.Image:
        """Turns an image and optionally a corresponding mask into a screenshot collage (2D image)"""
        # Generate all screenshots
        try:
            images = self._make_array_iterator(image)
        except ValueError as e:
            raise ValueError("Images need to be one or many numpy arrays") from e

        masks = self._make_array_iterator(mask, none_permitted=True)
        spacings = self._make_floats_iterator(spacing)
        captions = self._make_str_iterator(caption)

        try:
            screenshots = [
                [
                    g(next(images), next(masks), spacing=next(spacings), caption=next(captions))
                    for g in gg
                ]
                for gg in self.generators
            ]
        except StopIteration:
            raise ValueError(
                "Number of supplied images, masks and/or spacings does not match number of generators"
            )

        # Compute total width and height
        n_rows = len(screenshots)
        max_width = max(sum(s.width for s in ss) for ss in screenshots)
        max_cols = max(len(ss) for ss in screenshots)
        max_height = max(
            sum(ss[i].height for ss in screenshots if i < len(ss)) for i in range(max_cols)
        )

        content_width = max_width + (max_cols - 1) * self.spacing
        content_height = max_height + (n_rows - 1) * self.spacing

        width = content_width + 2 * self.margin
        height = content_height + 2 * self.margin

        # Generate new image with specified background color
        collage = PILImage.new(mode="RGB", size=(width, height), color=self.background)

        # Copy and paste screenshots into new image
        row_offset = self.margin
        for ss in screenshots:
            row_width = sum(s.width for s in ss) + (len(ss) - 1) * self.spacing
            col_offset = self.margin + round((content_width - row_width) / 2)

            for s in ss:
                collage.paste(s, (col_offset, row_offset))
                col_offset += s.width + self.spacing
            row_offset += max(s.height for s in ss) + self.spacing

        return collage

    @staticmethod
    def _make_array_iterator(
        arrays: Optional[CollageGrid[np.ndarray]], none_permitted: bool = False
    ) -> Iterator[np.ndarray]:
        if arrays is None and not none_permitted:
            raise ValueError("Numpy array expected, but None supplied")

        if arrays is None or isinstance(arrays, np.ndarray):
            return itertools.repeat(arrays)

        flattened = []
        for array in arrays:
            if array is None and not none_permitted:
                raise ValueError("Numpy array expected, but None supplied")

            if array is None or isinstance(array, np.ndarray):
                flattened.append(array)
            else:
                flattened.extend(array)

        return iter(flattened)

    @staticmethod
    def _make_floats_iterator(
        floats: Optional[CollageGrid[Iterable[float]]] = None,
    ) -> Iterator[Optional[Tuple[float]]]:
        if floats is None:
            return itertools.repeat(None)

        def _unwrap(iterable: Iterable) -> Union[list, tuple]:
            values = tuple(iterable)
            unwrapped = []

            for item in values:
                try:
                    iterator = iter(item)
                except TypeError:
                    return values
                else:
                    unwrapped_item = _unwrap(iterator)
                    if isinstance(unwrapped_item, list):
                        unwrapped.extend(unwrapped_item)
                    else:
                        unwrapped.append(unwrapped_item)

            return unwrapped

        flattened = _unwrap(floats)
        if isinstance(flattened, tuple):
            return itertools.repeat(flattened)
        else:
            return iter(flattened)

    @staticmethod
    def _make_str_iterator(
        strings: Optional[CollageGrid[str]] = None,
    ) -> Iterator[Optional[str]]:
        if strings is None or isinstance(strings, str):
            yield from itertools.repeat(strings)
        else:
            for s in strings:
                if isinstance(s, str):
                    yield s
                else:
                    yield from s


def find_center(image: np.ndarray, *, axis: int = 2, threshold: Union[int, float, bool] = 1):
    """Helper for finding an interesting region in the image"""
    center_of_mass = ndimage.center_of_mass(image >= threshold)
    absolute_coordinate = center_of_mass[axis]
    relative_coordinate = absolute_coordinate / (image.shape[axis] - 1)
    return relative_coordinate


def imshow(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    dpi: Optional[float] = None,
    axis_labels: bool = False,
    **screenshot_generator_args,
):
    """Helper for plotting a screenshot using matplotlib"""
    screenshot = ScreenshotGenerator(**screenshot_generator_args)
    plt.figure(dpi=dpi)
    plt.imshow(screenshot(image, mask), cmap="gray")

    if axis_labels:
        plt.gca().xaxis.ticK_top()
    else:
        plt.axis("off")
