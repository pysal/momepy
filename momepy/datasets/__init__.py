from pathlib import Path

__all__ = ["available", "get_path"]

_module_path = Path(__file__).resolve().parent
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
        return str(_module_path / (f"{dataset}.{extension}"))
    msg = (
        f"The dataset {dataset!r} is not available. "
        f"Available datasets are {', '.join(available)}"
    )
    raise ValueError(msg)
