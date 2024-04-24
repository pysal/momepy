import pandas as pd
import pytest
from pandas.testing import assert_index_equal


def assert_result(result, expected, geometry, **kwargs):
    """Check the expected values and types of the result."""
    for key, value in expected.items():
        assert getattr(result, key)() == pytest.approx(value)
    assert isinstance(result, pd.Series)
    assert_index_equal(result.index, geometry.index, **kwargs)
