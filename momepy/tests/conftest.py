import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal


def assert_result(result, expected, geometry, rel=None, abs=None, **kwargs):  # noqa: A002
    """Check the expected values and types of the result.
    Note: ''count'' refers to the number of non-NAs in the result."""
    for key, value in expected.items():
        assert getattr(result, key)() == pytest.approx(value, rel=rel, abs=abs)
    assert isinstance(result, pd.Series)
    assert_index_equal(result.index, geometry.index, **kwargs)


def assert_frame_result(result, expected, geometry, **kwargs):
    """Check the expected values and types of the result."""
    for key, value in expected.items():
        if key == "count":
            assert len(result) == pytest.approx(value)
        elif key == "mean":
            assert np.mean(result) == pytest.approx(value)
        elif key == "max":
            assert np.max(result) == pytest.approx(value)
        elif key == "min":
            assert np.min(result) == pytest.approx(value)
    assert_index_equal(result.index, geometry.index, **kwargs)
