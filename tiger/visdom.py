import csv
import re
import socket
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import numpy as np

try:
    from PIL import Image as PILImage
    from visdom import Visdom
except ImportError as e:
    raise ImportError(
        "The visdom and/or pillow packages are not installed but are required by the tiger.visdom module"
    ) from e

from .screenshots import DefaultColors


def _timestamp():
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def visdom_client(value=None):
    """Attempts to construct a visdom client from a port or url, or both"""
    if isinstance(value, Visdom):
        return value
    else:
        if not value:
            return Visdom()
        else:
            try:
                return Visdom(port=int(value))
            except ValueError:
                m = re.match("(.+):([0-9]+)", value)
                if m is None:
                    return Visdom(server=value)
                else:
                    return Visdom(server=m.group(1), port=int(m.group(2)))


def clear(client=None, env=None):
    """Closes all windows in an environment"""
    visdom = visdom_client(client)

    # Make sure that the environment exists by creating a dummy window
    visdom.text("Clearing environment", env=env)

    # Wait a bit before closing the environment, otherwise the dummy message is sometimes delayed
    # and the environment is deleted before the message is shown
    sleep(1)

    visdom.close(env=env)


class LearningCurve:
    """
    Plots multiple lines with Visdom by sending one data point per line each time.

    Parameters
    ----------
    title : string
        Caption of the graph

    legend : iterable
        Defines how many lines are plotted in the graph and how they are labeled

    moving_average : int
        Number of previous values to average (1 = no averaging)

    client : visdom.Visdom, string, int or None
        Visdom client instance. If not supplied, a new client with default hostname and port is created.
        If an integer is supplied, the value is interpreted as a port number. If a string is supplied,
        the value is interpreted as a host name.

    env : string or None
        Environment in visdom to which the data is send

    filename : pathlib.Path, string or None
        Path to a text / csv file in which the posted values are additionally saved

    **kwargs
        Additional arguments are forwarded to the options argument of Visdom.line
    """

    def __init__(
        self,
        title,
        legend=("Training", "Validation"),
        moving_average=1,
        client=None,
        env=None,
        filename=None,
        **kwargs,
    ):
        assert moving_average >= 1

        self.visdom = visdom_client(client)
        self.window = None

        legend = tuple(legend)
        self.num_lines = len(legend)
        self.submitted = 0

        self.moving_average = int(moving_average)
        self.previous_values = deque()

        self.env = env
        self.options = kwargs
        self.options["title"] = f"{title} ({_timestamp()})"
        self.options["legend"] = list(legend)
        self.options["layoutopts"] = {"plotly": {"title": title}}

        self.filename = Path(filename)
        if self.filename is not None:  # write header of CSV file
            with open(str(self.filename), "w") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(legend)

    def __len__(self):
        return self.submitted

    def post(self, values, step=None):
        """
        Adds a new data point to each line in the graph

        Parameters
        ----------
        values : numeric or iterable
            One new data point per line, expects exactly as many arguments as there are entries in the legend

        step : numeric
            Time point, such as the current epoch or number of minibatches (x-axis). If not value is supplied,
            the internal counter is used.

        Raises
        ------
        ValueError
            If an incorrect number of values is supplied

        IOError
            If writing the values to a CSV file (if a filename was supplied to the constructor) fails
        """
        y = np.asarray(values).flatten()
        if y.size != self.num_lines:
            raise ValueError(f"Expected {self.num_lines} values, but got {y.size} values")

        # Append values to CSV file
        if self.filename is not None:
            with open(str(self.filename), "a") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(v) for v in y])

        # Compute moving average if that is enabled
        if self.moving_average > 1:
            self.previous_values.append(y)
            if len(self.previous_values) < self.moving_average:
                return
            elif len(self.previous_values) > self.moving_average:
                self.previous_values.popleft()

            y = np.mean(self.previous_values, axis=0)

        # Submit values to visdom server
        if step is None:
            step = self.submitted + self.moving_average

        if self.num_lines == 1:
            x = np.asarray(step).reshape(1)
        else:
            x = np.repeat(step, self.num_lines).reshape(1, -1)
            y = y.reshape(1, -1)

        if self.window:
            self.visdom.line(
                X=x,
                Y=y,
                win=self.window,
                env=self.env,
                opts=self.options,
                update="append",
            )
        else:
            self.window = self.visdom.line(X=x, Y=y, env=self.env, opts=self.options)

        self.submitted += 1


class Image:
    """
    Displays an image, optionally with overlay.

    Parameters
    ----------
    title : string
        Caption of the image

    scale : bool
        If scale is True, the image is rescaled to fit into [0,255]

    overlay_opacity : float
        Opacity of the overlay

    overlay_lut : callable or None
        Color LUT (callable object) that returns RGBA colors for labels

    client : visdom.Visdom, string, int or None
        Visdom client instance. If not supplied, a new client with default hostname and port is created.
        If an integer is supplied, the value is interpreted as a port number. If a string is supplied,
        the value is interpreted as a host name.

    env : string or None
        Environment in visdom to which the data is send

    **kwargs
        Additional arguments are forwarded to the options argument of Visdom.image
    """

    def __init__(
        self,
        title,
        scale=True,
        overlay_opacity=0.6,
        overlay_lut=None,
        client=None,
        env=None,
        **kwargs,
    ):
        self.visdom = visdom_client(client)
        self.window = None

        self.env = env
        self.options = kwargs
        self.options["title"] = f"{title} ({_timestamp()})"

        self.scale = scale
        self.overlay_opacity = overlay_opacity
        self.labels_to_colors = np.vectorize(
            DefaultColors() if overlay_lut is None else overlay_lut
        )

    def apply_opacity(self, alpha):
        return np.clip(np.round(alpha * self.overlay_opacity), 0, 255).astype("uint8")

    def post(self, image, overlay=None, filename=None):
        """
        Sends an image (optionally with an overlay) to the visdom server. If another image has been send before,
        the image pane is updated with the new image.

        Parameters
        ----------
        image : ndarray
            Either a 2D (WxH) matrix or a 3D matrix (WxHxC)

        overlay : ndarray or None
            Can be a 2D (WxH) matrix with integer labels or a 3D matrix (WxHxC) with RGB/RGBA colors. For RGBA overlays,
            the value of "overlay_opacity" is ignored.

        filename : pathlib.Path, string or None
            If a filename is given, the image with overlayed mask is additionally saved to that file
        """
        # Prepare the image
        image = np.asarray(image)
        if self.scale:
            image = image.astype(float)
            image -= np.min(image)
            max_value = np.max(image)
            if max_value < 1e-8:
                image = np.zeros_like(image)
            else:
                image = np.clip(np.round(image / max_value * 255), 0, 255)
        image = image.astype("uint8")

        # Add the overlay to the image
        if overlay is None or self.overlay_opacity == 0:
            if image.ndim == 3:
                data = np.transpose(image, (2, 1, 0))
            else:
                data = np.transpose(image, (1, 0))
        else:
            overlay = np.asarray(overlay)

            if overlay.shape[:2] != image.shape[:2]:
                raise ValueError(
                    "Image and overlay dimensions are different, got data of shape {} and {}".format(
                        image.shape, overlay.shape
                    )
                )

            if overlay.ndim == 2:
                # Turn labels into RGBA colors and adjust alpha value
                overlay = np.transpose(self.labels_to_colors(overlay), (1, 2, 0)).astype("uint8")
                overlay[:, :, 3] = self.apply_opacity(overlay[:, :, 3])
            elif overlay.ndim == 3:
                # Overlay is already RGB or RGBA
                overlay = overlay.astype("uint8")

                if overlay.shape[2] == 4:
                    # Assume the overlay is RGBA, nothing to do
                    pass
                elif overlay.shape[2] == 3:
                    # Assume the overlay is RGB, add alpha channel
                    alpha = np.ones(shape=overlay.shape[:2], dtype="uint8") * self.apply_opacity(
                        255
                    )
                    alpha[np.sum(overlay, axis=2) == 0] = 0
                    overlay = np.concatenate([overlay, alpha[:, :, np.newaxis]], axis=2)
                else:
                    raise ValueError("Overlay has unknown data format")
            else:
                raise ValueError(
                    f"Overlay needs to be HxW or HxWxC, got {overlay.ndim} dimensional array instead"
                )

            # Place the overlay on top of the image
            background = PILImage.fromarray(image).convert("RGBA")
            foreground = PILImage.fromarray(overlay, mode="RGBA")
            composition = PILImage.alpha_composite(background, foreground).convert("RGB")
            data = np.transpose(np.asarray(composition), (2, 1, 0))  # visdom needs CxHxW

            if filename is not None:
                composition.save(str(filename))

        # Send the image to the visdom server
        if self.window:
            self.visdom.image(data, win=self.window, env=self.env, opts=self.options)
        else:
            self.window = self.visdom.image(data, env=self.env, opts=self.options)


class RuntimeMonitor:
    """Monitors the runtime over multiple iterations and plots them as continous graph"""

    def __init__(
        self,
        title="Runtime per iteration",
        average=5,
        start=False,
        client=None,
        env=None,
        **kwargs,
    ):
        assert average >= 1

        self.visdom = visdom_client(client)
        self.window = None

        self.average = int(average)
        self.cache = None
        self.timestamps = []
        self.runtimes = []

        options = kwargs
        options["title"] = f"{title} ({_timestamp()})"

        # Construct payload to send to the visdom server
        self.payload = {
            "data": [
                {"x": self.timestamps, "y": self.runtimes, "type": "scatter", "name": "runtime"}
            ],
            "name": "runtime",
            "opts": options,
            "layout": {"title": title},
            "eid": env,
        }

        if start:
            self.start()

    def __len__(self):
        return len(self.timestamps) * self.average

    def start(self):
        """Starts monitoring the runtime, call right before beginning the first iteration"""
        self.cache = [datetime.now()]

    def step(self):
        """Records the current timestamp as the end of an iteration and begin of the next iteration"""
        if self.cache is None:
            raise RuntimeError("Call start() before the first step")

        self.cache.append(datetime.now())

        if len(self.cache) > self.average:
            # Calculate average runtime
            delta = self.cache[-1] - self.cache[0]
            midpoint = self.cache[0] + delta / 2
            runtime = delta.total_seconds() / self.average

            self.timestamps.append(str(midpoint))
            self.runtimes.append(runtime)
            self.cache = self.cache[-1:]

            # Send all values to the server (again)
            if self.window:
                self.payload["win"] = self.window
            self.window = self.visdom._send(self.payload)


class TextBox:
    """Displays some text in large letters in the visdom environment"""

    def __init__(self, title, font_size=38, width=220, height=80, client=None, env=None, **kwargs):
        self.visdom = visdom_client(client)
        self.window = None
        self.env = env

        self.font_size = int(font_size)

        self.options = kwargs
        self.options.update(width=width, height=height, title=f"{title} ({_timestamp()})")

    def post(self, text):
        """Updates the displayed text"""
        html = (
            '<span style="font-size: {}px; display: block; text-align: center;">{}</span>'.format(
                self.font_size, text
            )
        )
        if self.window:
            self.visdom.text(html, win=self.window, env=self.env, opts=self.options)
        else:
            self.window = self.visdom.text(html, env=self.env, opts=self.options)


class Epoch(TextBox):
    """Displays the current epoch in large letters in the visdom environment"""

    def __init__(
        self,
        title="Epoch",
        font_size=38,
        width=220,
        height=80,
        client=None,
        env=None,
        **kwargs,
    ):
        super().__init__(title, font_size, width, height, client, env, **kwargs)
        self.last_epoch = 0

    def post(self, epoch=None):
        """Updates the displayed epoch number"""
        if epoch is None:
            epoch = self.last_epoch + 1
        super().post(epoch)
        self.last_epoch = epoch


class Hostname(TextBox):
    """Displays the name of the machine on which the network is training"""

    def __init__(
        self,
        title="Hostname",
        font_size=38,
        width=220,
        height=80,
        client=None,
        env=None,
        **kwargs,
    ):
        super().__init__(title, font_size, width, height, client, env, **kwargs)

    def post(self, hostname=None):
        """Updates the displayed host name"""
        if hostname is None:
            hostname = socket.gethostname().lower()
        super().post(hostname)


def post_hostname(**kwargs):
    """Posts the name of the current machine (provides a shortcut for creating a Hostname instance)"""
    h = Hostname(**kwargs)
    h.post()
    del h  # not actually needed due to garbage collection


class RemainingTime(TextBox):
    """Calculates and displays the remaining time using an instance of the runtime monitor"""

    def __init__(
        self,
        epochs,
        title="Remaining time",
        font_size=38,
        width=450,
        height=130,
        client=None,
        env=None,
        **kwargs,
    ):
        assert epochs > 0

        super().__init__(title, font_size, width, height, client, env, **kwargs)
        self.epochs = int(epochs)
        self.tfoi = max(10, int(round(self.epochs * 0.00025)))
        self.last_epoch = 0

    def post(self, runtime_monitor, epoch=None):
        if isinstance(epoch, Epoch):
            epoch = epoch.last_epoch
        elif epoch is None:
            epoch = self.last_epoch + 1

        if len(runtime_monitor.runtimes) * runtime_monitor.average >= self.tfoi:
            # Calculate remaining time based on average runtime over last few epochs
            runtimes = runtime_monitor.runtimes[-self.tfoi :]
            runtime_per_epoch = np.mean(runtimes)
            remaining_runtime = (self.epochs - epoch) * runtime_per_epoch

            # Use datetime's timedelta to format the remaining time nicely
            text = "{}<br>{:.1f} &plusmn; {:.1f} s/it".format(
                timedelta(seconds=round(remaining_runtime)),
                runtime_per_epoch,
                np.std(runtimes),
            )
            super().post(text)

        self.last_epoch = epoch
