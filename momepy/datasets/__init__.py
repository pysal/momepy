from pathlib import Path

__all__ = ["available", "get_path"]

_module_path = Path(__file__).resolve().parent
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
        return str(_module_path / (dataset + ".gpkg"))
    msg = (
        f"The dataset {dataset!r} is not available. "
        f"Available datasets are {', '.join(available)}"
    )
    raise ValueError(msg)
