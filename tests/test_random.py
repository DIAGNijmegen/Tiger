import tiger.random

# Unit testing functions that return pseudo-random values is difficult, so
# we test here using a fixed random seed and multiple repetitions and make
# some statistical assumptions (should not return the same number every time
# when repeated often enough, etc)
tiger.random.seed(0)


def test_random_index():
    items = [1, 2, 3, 4, 5]

    indices = set()
    for _ in range(100):
        index = tiger.random.random_index(items)
        assert isinstance(index, int) and 0 <= index < len(items)
        indices.add(index)

    assert len(indices) == len(items)


def test_random_item():
    items = [1, 2, 3, 4, 5]

    picked = set()
    for _ in range(100):
        item = tiger.random.random_item(items)
        assert item in items
        picked.add(item)

    assert picked == set(items)


def test_random_decision():
    # When sampling with 50% probability True vs False, we can expect
    # approximately 50 +/- 10 times True when drawing 100 samples
    returned_values = [tiger.random.random_decision(0.5) for _ in range(100)]
    assert set(returned_values) == {True, False}
    assert (sum(1 for p in returned_values if p is True) - 50) < 10

    # Sample with 0% = only False is returned
    returned_values = {tiger.random.random_decision(0) for _ in range(100)}
    assert returned_values == {False}

    # Sample with 100% = only True is returned
    returned_values = {tiger.random.random_decision(1) for _ in range(100)}
    assert returned_values == {True}
