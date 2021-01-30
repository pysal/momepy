#!/usr/bin/env python
# -*- coding: utf-8 -*-

import libpysal
import numpy as np

__all__ = ["DistanceBand", "sw_high"]


class DistanceBand:
    """
    On demand distance-based spatial weights-like class.

    Mimic the behavior of ``libpysal.weights.DistanceBand`` but do not compute all
    neighbors at once but only on demand. Only ``DistanceBand.neighbors[key]`` is
    implemented. Once user asks for ``DistanceBand.neighbors[key]``, neighbors for
    specified key will be computed using rtree. The algorithm is significantly
    slower than ``libpysal.weights.DistanceBand`` but allows for large number of
    neighbors which may cause memory issues in libpysal.

    Use ``libpysal.weights.DistanceBand`` if possible. ``momepy.weights.DistanceBand``
    only when necessary. ``DistanceBand.neighbors[key]`` should yield same results as
    :class:`momepy.DistanceBand`.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        GeoDataFrame containing objects to be used
    threshold : float
        distance band to be used as buffer
    centroid : bool (default True)
        use centroid of geometry (as in ``libpysal.weights.DistanceBand``).
        If ``False``, works with the geometry as it is.
    ids : str
        column to be used as geometry ids. If not set, integer position is used.

    Attributes
    ----------
    neighbors[key] : list
        list of ids of neighboring features

    """

    def __init__(self, gdf, threshold, centroid=True, ids=None):
        if centroid:
            gdf = gdf.copy()
            gdf.geometry = gdf.centroid

        self.neighbors = _Neighbors(gdf, threshold, ids=ids)

    def fetch_items(self, key):
        hits = self.sindex.query(self.bufferred[key], predicate="intersects")
        match = self.geoms.iloc[hits].index.to_list()
        match.remove(key)
        return match


class _Neighbors(dict, DistanceBand):
    """
    Helper class for DistanceBand.
    """

    def __init__(self, geoms, buffer, ids):
        self.geoms = geoms
        self.sindex = geoms.sindex
        self.bufferred = geoms.buffer(buffer)
        if ids:
            self.ids = np.array(geoms[ids])
            self.ids_bool = True
        else:
            self.ids = range(len(self.geoms))
            self.ids_bool = False

    def __missing__(self, key):
        if self.ids_bool:
            int_id = np.where(self.ids == key)[0][0]
            integers = self.fetch_items(int_id)
            return list(self.ids[integers])
        else:
            return self.fetch_items(key)

    def keys(self):
        return self.ids


def sw_high(k, gdf=None, weights=None, ids=None, contiguity="queen", silent=True):
    """
    Generate spatial weights based on Queen or Rook contiguity of order k.

    Adjacent are all features within <= k steps. Pass either ``gdf`` or ``weights``.
    If both are passed, ``weights`` is used.
    If ``weights`` are passed, ``contiguity`` is
    ignored and high order spatial weights based on ``weights`` are computed.

    Parameters
    ----------
    k : int
        order of contiguity
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse. Index has to be
        consecutive range ``0:x``.
        Otherwise, spatial weights will not match objects.
    weights : libpysal.weights
        libpysal.weights of order 1
    contiguity : str (default 'queen')
        type of contiguity weights. Can be ``'queen'`` or ``'rook'``.
    silent : bool (default True)
        silence libpysal islands warnings

    Returns
    -------
    libpysal.weights
        libpysal.weights object

    Examples
    --------
    >>> first_order = libpysal.weights.Queen.from_dataframe(geodataframe)
    >>> first_order.mean_neighbors
    5.848032564450475
    >>> fourth_order = sw_high(k=4, gdf=geodataframe)
    >>> fourth.mean_neighbors
    85.73188602442333

    """
    if weights is not None:
        first_order = weights
    elif gdf is not None:
        if contiguity == "queen":
            first_order = libpysal.weights.Queen.from_dataframe(
                gdf, ids=ids, silence_warnings=silent
            )
        elif contiguity == "rook":
            first_order = libpysal.weights.Rook.from_dataframe(
                gdf, ids=ids, silence_warnings=silent
            )
        else:
            raise ValueError(f"{contiguity} is not supported. Use 'queen' or 'rook'.")
    else:
        raise AttributeError("GeoDataFrame or spatial weights must be given.")

    if k > 1:
        id_order = first_order.id_order
        w = first_order.sparse
        wk = sum(map(lambda x: w ** x, range(2, k + 1)))
        rk, ck = wk.nonzero()
        sk = set(zip(rk, ck))
        sk = set([(i, j) for i, j in sk if i != j])
        d = dict([(i, []) for i in id_order])
        for pair in sk:
            k, v = pair
            k = id_order[k]
            v = id_order[v]
            d[k].append(v)
        return libpysal.weights.W(neighbors=d, silence_warnings=silent)
    return first_order
