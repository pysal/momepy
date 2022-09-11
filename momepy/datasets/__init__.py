import os

__all__ = ["available", "get_path"]

_module_path = os.path.dirname(__file__)
available = ["bubenec", "tests", "nyc_graph"]


def get_path(dataset, extension="gpkg"):
    """
    Get the path to the data file.
    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``momepy.datasets.available`` for
        all options.
    extension : str
        The extension of the data file
    """
    if dataset in available:
        filepath = dataset + "." + extension
        return os.path.abspath(os.path.join(_module_path, filepath))
    msg = "The dataset '{data}' is not available. ".format(data=dataset)
    msg += "Available datasets are {}".format(", ".join(available))
    raise ValueError(msg)
