import os

__all__ = ["available", "get_path"]

_module_path = os.path.dirname(__file__)
available = ["bubenec", "tests"]


def get_path(dataset):
    """
    Get the path to the data file.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``momepy.datasets.available`` for
        all options.
    """
    if dataset in available:
        return os.path.abspath(os.path.join(_module_path, dataset + ".gpkg"))
    msg = f"The dataset '{dataset}' is not available. "
    msg += "Available datasets are {}".format(", ".join(available))
    raise ValueError(msg)
