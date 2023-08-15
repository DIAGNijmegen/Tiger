import itertools
import os
import sys
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np

# Define generic type for type hinting
T = TypeVar("T")


def repeat_if_needed(x: Optional[Union[T, Iterable[T]]], n: int) -> Tuple[T, ...]:
    """Converts value x into a tuple and ensures that the length is n

    If x is a non-iterable value, like int or float, it will be repeated n times.
    If x is an iterable value, like a list, it will be checked whether it contains n items.

    Parameters
    ----------
    x
        A single value or an iterable with n elements (numbers)
    n
        Number of expected elements
    t
        Type of the elements, will be used to convert all elements to the target type

    Returns
    -------
    tuple
        A tuple with n elements of type t

    Raises
    ------
    ValueError
        If x is None and return_none is False, or if x is a sequence of fewer or more than n elements
    """
    if x is None:
        raise ValueError(
            f"Expected a single value or an iterable with {n} elements, got None instead"
        )

    try:
        y = tuple(x)  # type: ignore
        if len(y) != n:
            raise ValueError(
                f"Expected a single value or an iterable with {n} elements, got {x} instead"
            )
        else:
            return y
    except TypeError:
        return tuple(itertools.repeat(x, n))  # type: ignore


def slice_nd_array(
    slice_shape: Iterable[int], offset: Optional[Iterable[int]] = None
) -> Tuple[slice, ...]:
    """Get a numpy slice for an array of n dimensions using an offset.

    Parameters
    ----------
    slice_shape
        Shape of the mask.
    offset
        Offset with respect to the reference image in which slicing is used.

    Returns
    -------
    indices
        A slice object to select a mask in an image.
    """
    slice_shape = tuple(slice_shape)
    if offset is None:
        offset = itertools.repeat(0, len(slice_shape))

    return tuple(np.s_[o : o + s] for s, o in zip(slice_shape, offset))


@contextmanager
def suppress_output(stream: str = "stdout"):
    """Suppress all output to stdout or another standard stream

    Does currently not work under Windows. Suppresses really all output, so also that of c-modules.

    Parameters
    ----------
    stream
        The standard output stream that should be silenced. Defaults to stdout, but can also be "stderr"

    Examples
    --------
    >>> with suppress_output():
    >>>    print("Foo")

    """
    if stream not in ("stdout", "stderr"):
        raise ValueError(
            f'Unknown stream name "{stream}", can only suppress outout of "stdout" or "stderr"'
        )

    with open(os.devnull, "w") as devnull:
        jupyter = False

        try:
            # Store a copy of the original stream handle, overwrite with devnull handle
            original_stream = sys.stdout if stream == "stdout" else sys.stderr
            stream_handle = original_stream.fileno()
            original_stream_handle = os.dup(stream_handle)
            os.dup2(devnull.fileno(), stream_handle)
        except OSError:
            # Failure replacing the stream, can happen in jupyter notebooks for example because the
            # jupyter kernel replaces the output streams in sys to capture the output - so we need
            # to replace jupyter's replacement temporarily
            jupyter = True

            stream_handle = getattr(sys, stream)
            setattr(sys, stream, devnull)

        # Do action, eventually restore original stream handle
        try:
            yield
        finally:
            if not jupyter:
                os.dup2(original_stream_handle, stream_handle)
            else:
                setattr(sys, stream, stream_handle)


def first(iterable: Iterable[T], condition: Optional[Callable[[T], bool]] = None) -> T:
    """Returns the first object for an iterable, replaces next(iter(...)) constructions"""
    for item in iterable:
        if condition is None or condition(item):
            return item
    raise ValueError("Iterable did not yield a single suitable object")


def count(iterable: Iterable[T], condition: Optional[Callable[[T], bool]] = None) -> int:
    """Returns the number of items that evaluate to True in the iterable object"""
    if condition is None:
        return sum(1 for v in iterable if v)
    else:
        return sum(1 for v in iterable if condition(v))
