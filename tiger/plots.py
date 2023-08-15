import csv
from pathlib import Path
from typing import Iterable, List, Optional, OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from .io import PathLike, read_json, write_json


class LearningCurve:
    """Helps keep track of several metrics while training a machine learning algorithm

    The class serves both as a container in which the loss values etc can be collected
    over time, but also offers quick and easy ways to save the raw values as well as
    plots of the learning curve. This makes it easy to either continue training later
    and restore the previous values from a json or csv file, or to load the values after
    the training is complete in order to plot the learning curve. During training, the
    class enables regular plotting and saving of a learning curve as an image.

    Parameters
    ----------
    metrics
        The names of the different metrics, e.g., `['Training loss', 'Validation loss']`
        Mandatory when not loading from a file.

    file
        Path to a json or csv file. If the file exists, the learning curve is populated with
        values from that file.

    auto_save
        Flag that indicates whether appending values automatically triggers an update of the
        underlying csv or json file. Ignored if filename was not set.
    """

    def __init__(
        self,
        *,
        metrics: Optional[Iterable[str]] = None,
        file: Optional[PathLike] = None,
        auto_save: bool = False,
        auto_plot: bool = False,
    ):
        self.metrics: List[str] = []
        self.values = OrderedDict[str, List[float]]()
        self.file = None if file is None else Path(file)
        self.auto_save = auto_save
        self.auto_plot = auto_plot
        self.cursor = 0

        if self.file is None or not self.file.exists():
            # Start fresh, there is no file that we can load previous values from
            if metrics is None:
                raise ValueError("Expected at least one metric, got None")

            self.metrics.extend(metrics)
            if len(self.metrics) == 0:
                raise ValueError("Expected at least one metric, got none")

            for metric in self.metrics:
                self.values[metric] = []
        else:
            # Load data from file
            if self.file.suffix == ".json":
                snapshot = read_json(self.file)
                for metric, values in snapshot.items():
                    self.metrics.append(metric)
                    self.values[metric] = values
            elif self.file.suffix == ".csv":
                with self.file.open(encoding="UTF-8") as csvfile:
                    for i, row in enumerate(csv.reader(csvfile)):
                        if i == 0:
                            for metric in row:
                                self.metrics.append(metric.strip())
                                self.values[metric] = []
                        else:
                            for metric, value in zip(self.metrics, row):
                                self.values[metric].append(float(value.strip()))
                            self.cursor += 1
            else:
                raise ValueError(
                    "Only JSON and CSV files are supported, and filenames must have appropriate extensions"
                )

            if metrics is not None and self.metrics != list(metrics):
                raise ValueError("Metrics in file are different from specified metrics")

    def __len__(self):
        """Number of data points"""
        return len(next(iter(self.values.values())))

    def append(self, values: Iterable[float]):
        """Appends new values

        Parameters
        ----------
        values
            One value per metric
        """
        for metric, value in zip(self.metrics, values):
            self.values[metric].append(value)

        if self.auto_save:
            self.save()
        if self.auto_plot:
            self.plot(None if self.file is None else self.file.with_suffix(".png"))

    def save(self, file: Optional[PathLike] = None):
        """Saves data to a json or csv file

        When writing to a JSON file, the entire file is replaced. When writing to a CSV file, only
        new values are appended, unless a new filename is provided.

        Parameters
        ----------
        file
            Output file, overwrites self.file and is required if no filename has previously been set

        Raises
        ------
        ValueError
            If no filename was provided or previously configured, or if the filename does not point to
            either a json or csv file.
        """
        if file is not None:
            self.file = Path(file)
            self.cursor = 0
        elif self.file is None:
            raise ValueError("Cannot save learning curve, output file not configured")

        if self.file.suffix == ".json":
            write_json(self.file, self.values)
        elif self.file.suffix == ".csv":
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with self.file.open("a", newline="\n", encoding="UTF-8") as csvfile:
                append = csvfile.tell() > 0
                n_values = len(self)

                writer = csv.writer(csvfile)
                if append:
                    # Write only new values
                    for i in range(self.cursor, n_values):
                        writer.writerow([str(self.values[metric][i]) for metric in self.metrics])
                else:
                    # Write header
                    writer.writerow(self.metrics)

                    # Write all values
                    writer.writerows(
                        [
                            [str(self.values[metric][i]) for metric in self.metrics]
                            for i in range(n_values)
                        ]
                    )

                self.cursor = n_values
        else:
            raise ValueError(
                "Only JSON and CSV files are supported, and filenames must have appropriate extensions"
            )

    def plot(
        self,
        file: Optional[PathLike] = None,
        *,
        running_mean: int = 0,
        title: Optional[str] = None,
        metrics: Optional[Iterable[str]] = None,
    ):
        """Generates a learning curve plot

        Parameters
        ----------
        file
            Path where the plot is saved as image (should have an extension like png or jpg). If no filename
            is provided, the plot is instead shown in a matplotlib window.

        running_mean
            Number of data points that are averaged (>= 0). Values 0 or 1 correspond to no averaging.

        title
            A string such as "Learning curve" that is added to the top of the figure.

        metrics
            List of metrics to include in the plot, otherwise all are included.
        """
        # Prepare running mean computation
        if running_mean > 1:
            n = running_mean
            running_mean = True
            running_mean_kernel = np.ones(n) / n
        else:
            running_mean = False
            running_mean_kernel = None

        # Plot the individual curves
        plt.figure(figsize=(16, 8))
        included_metrics = set(self.metrics) if metrics is None else set(metrics)
        for metric in self.metrics:
            if metric not in included_metrics:
                continue

            values = self.values[metric]
            if running_mean and len(values) > len(running_mean_kernel):
                values = np.convolve(np.asarray(values), running_mean_kernel, mode="valid")

            plt.plot(values, label=metric)

        # Add legend etc
        plt.legend()
        if title:
            plt.title(title)
        plt.tight_layout(pad=1.2)

        # Show plot or save to file?
        if file is None:
            plt.show()
        else:
            file = Path(file)
            file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(file), dpi=160)
            plt.close()

    @staticmethod
    def load_and_plot(data_file: PathLike, image_file: Optional[PathLike] = None, **plot_args):
        """Helper function to load and immediately plot a learning curve from a file"""
        learning_curve = LearningCurve(file=data_file)
        learning_curve.plot(file=image_file, **plot_args)
