from typing import Optional, Sequence, Sized, TypeVar

import numpy as np

# Create type variable for generic typing
T = TypeVar("T")

# Create a random number generator for use by the functions below
_random_state = np.random.RandomState()


def seed(seed_value: Optional[int] = None):
    """The helper functions in this module are based on numpy.random"""
    _random_state.seed(seed_value)


def random_index(indexable: Sized) -> int:
    """Returns a random index from a list"""
    return _random_state.randint(0, len(indexable))


def random_item(indexable: Sequence[T]) -> T:
    """Returns a single random item from a list"""
    return _random_state.choice(indexable)


def random_decision(probability: float) -> bool:
    """Returns True with the given probability"""
    return _random_state.random_sample() <= probability
