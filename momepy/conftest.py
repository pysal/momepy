import geopandas
import pytest

import momepy


@pytest.fixture(autouse=True)
def add_momepy_and_geopandas(doctest_namespace):
    doctest_namespace["momepy"] = momepy
    doctest_namespace["geopandas"] = geopandas
