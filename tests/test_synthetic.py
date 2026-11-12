import numpy as np
import pytest

from rf_sensing.synthetic import make_dataset


def test_generator_is_seeded_and_returns_iq_channels():
    first_x, first_y = make_dataset(16, samples_per_window=64, seed=123)
    second_x, second_y = make_dataset(16, samples_per_window=64, seed=123)
    assert first_x.shape == (16, 2, 64)
    assert first_x.dtype == np.float32
    assert first_y.shape == (16,)
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


def test_generator_rejects_invalid_size():
    with pytest.raises(ValueError):
        make_dataset(0)
