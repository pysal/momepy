__all__ = [
    "simpson_diversity",
    "shannon_diversity",
]


def simpson_diversity(values, bins=None, categorical=False):
    """
    Calculates the Simpson's diversity index of data. Helper function for
    :py:class:`momepy.Simpson`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Parameters
    ----------
    values : pandas.Series
        A list of values.
    bins : array, optional
        An array of top edges of classification bins.
        Should be equal to the result of ``binning.bins``.
    categorical : bool (default False)
        Treat values as categories (will not use ``bins``).

    Returns
    -------
    float
        Simpson's diversity index.

    See also
    --------
    momepy.Simpson : Calculates the Simpson's diversity index.
    """
    if not categorical:
        try:
            import mapclassify as mc
        except ImportError as err:
            raise ImportError("The 'mapclassify' package is required") from err

    if categorical:
        counts = values.value_counts()

    else:
        sample_bins = mc.UserDefined(values, bins)
        counts = sample_bins.counts

    return sum((n / sum(counts)) ** 2 for n in counts if n != 0)


def shannon_diversity(data, bins=None, categorical=False, categories=None):
    """
    Calculates the Shannon's diversity index of data. Helper function for
    :py:class:`momepy.Shannon`.

    .. math::

        \\lambda=\\sum_{i=1}^{R} p_{i}^{2}

    Formula adapted from https://gist.github.com/audy/783125.

    Parameters
    ----------
    data : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    bins : array, optional
        An array of top edges of classification bins. Result of ``binning.bins``.
    categorical : bool (default False)
        tTeat values as categories (will not use ``bins``).
    categories : list-like (default None)
        A list of categories.

    Returns
    -------
    float
        Shannon's diversity index.

    See also
    --------
    momepy.Shannon : Calculates the Shannon's diversity index.
    momepy.Simpson : Calculates the Simpson's diversity index.
    momepy.simpson_diversity : Calculates the Simpson's diversity index.
    """
    from math import log as ln

    if not categorical:
        try:
            import mapclassify as mc
        except ImportError as err:
            raise ImportError("The 'mapclassify' package is required") from err

    def p(n, sum_n):
        """Relative abundance"""
        if n == 0:
            return 0
        return (n / sum_n) * ln(n / sum_n)

    if categorical:
        counts = dict.fromkeys(categories, 0)
        counts.update(data.value_counts())
    else:
        sample_bins = mc.UserDefined(data, bins)
        counts = dict(zip(bins, sample_bins.counts, strict=True))

    return -sum(p(n, sum(counts.values())) for n in counts.values() if n != 0)
