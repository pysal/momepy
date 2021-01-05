#!/usr/bin/env python
# -*- coding: utf-8 -*-

# elements.py
# generating derived elements (street edge, block)
import geopandas as gpd
import libpysal
import numpy as np
import pygeos
import pandas as pd
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize
from tqdm import tqdm

# TODO: this should not be needed with shapely 2.0
from geopandas._vectorized import _pygeos_to_shapely

__all__ = [
    "buffered_limit",
    "Tessellation",
    "Blocks",
    "get_network_id",
    "get_node_id",
    "enclosures",
    "get_network_ratio",
]


def buffered_limit(gdf, buffer=100):
    """
    Define limit for :class:`momepy.Tessellation` as a buffer around buildings.

    See :cite:`fleischmann2020` for details.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints
    buffer : float
        buffer around buildings limiting the extend of tessellation

    Returns
    -------
    MultiPolygon
        MultiPolygon or Polygon defining the study area

    Examples
    --------
    >>> limit = mm.buffered_limit(buildings_df)
    >>> type(limit)
    shapely.geometry.polygon.Polygon

    """
    study_area = gdf.copy()
    study_area["geometry"] = study_area.buffer(buffer)
    built_up = study_area.geometry.unary_union
    return built_up


class Tessellation:
    """
    Generates tessellation.

    Three versions of tessellation can be created:

    1. Morphological tessellation around given buildings ``gdf`` within set ``limit``.
    2. Proximity bands around given street network ``gdf`` within set ``limit``.
    3. Enclosed tessellation based on given buildings ``gdf`` within ``enclosures``.

    Pass either ``limit`` to create morphological tessellation or proximity bands or
    ``enclosures`` to create enclosed tessellation.

    See :cite:`fleischmann2020` for details of implementation of morphological
    tessellation and :cite:`araldi2019` for proximity bands.

    Tessellation requires data of relatively high level of precision and there are three
    particular patterns causing issues.\n
    1. Features will collapse into empty polygon - these do not have tessellation
    cell in the end.\n
    2. Features will split into MultiPolygon - at some cases, features with narrow links
    between parts split into two during 'shrinking'. In most cases that is not an issue
    and resulting tessellation is correct anyway, but sometimes this result in a cell
    being MultiPolygon, which is not correct.\n
    3. Overlapping features - features which overlap even after 'shrinking' cause
    invalid tessellation geometry.\n
    All three types can be tested prior :class:`momepy.Tessellation` using
    :class:`momepy.CheckTessellationInput`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints or street network
    unique_id : str
        name of the column with unique id
    limit : MultiPolygon or Polygon (default None)
        MultiPolygon or Polygon defining the study area limiting
        morphological tessellation or proximity bands
        (otherwise it could go to infinity).
    shrink : float (default 0.4)
        distance for negative buffer to generate space between adjacent polygons
        (if geometry type of gdf is (Multi)Polygon).
    segment : float (default 0.5)
        maximum distance between points after discretization
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
    enclosures : GeoDataFrame (default None)
        Enclosures geometry. Can  be generated using :func:`momepy.enclosures`.
    enclosure_id : str (default 'eID')
        name of the enclosure_id containing unique identifer for each row in ``enclosures``.
        Applies only if ``enclosures`` are passed.
    threshold : float (default 0.05)
        The minimum threshold for a building to be considered within an enclosure.
        Threshold is a ratio of building area which needs to be within an enclosure to
        inlude it in the tessellation of that enclosure. Resolves sliver geometry
        issues. Applies only if ``enclosures`` are passed.
    use_dask : bool (default True)
        Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        Applies only if ``enclosures`` are passed.
    n_chunks : None
        Number of chunks to be used in parallelization. Ideal is one chunk per thread.
        Applies only if ``enclosures`` are passed. Defualt automatically uses
        n == dask.system.cpu_count.

    Attributes
    ----------
    tessellation : GeoDataFrame
        GeoDataFrame containing resulting tessellation

        For enclosed tessellation, gdf contains three columns:
            - ``geometry``,
            - ``unique_id`` matching with parental building,
            - ``enclosure_id`` matching with enclosure integer index

    gdf : GeoDataFrame
        original GeoDataFrame
    id : Series
        Series containing used unique ID
    limit : MultiPolygon or Polygon
        limit
    shrink : float
        used shrink value
    segment : float
        used segment value
    collapsed : list
        list of unique_id's of collapsed features (if there are some)
        Applies only if ``limit`` is passed.
    multipolygons : list
        list of unique_id's of features causing MultiPolygons (if there are some)
        Applies only if ``limit`` is passed.

    Examples
    --------
    >>> tess = mm.Tessellation(
    ... buildings_df, 'uID', limit=mm.buffered_limit(buildings_df)
    ... )
    Inward offset...
    Generating input point array...
    Generating Voronoi diagram...
    Generating GeoDataFrame...
    Dissolving Voronoi polygons...
    >>> tess.tessellation.head()
        uID	geometry
    0	1	POLYGON ((1603586.677274485 6464344.667944215,...
    1	2	POLYGON ((1603048.399497852 6464176.180701573,...
    2	3	POLYGON ((1603071.342637536 6464158.863329805,...
    3	4	POLYGON ((1603055.834005827 6464093.614718676,...
    4	5	POLYGON ((1603106.417554705 6464130.215958447,...

    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
    >>> encl_tess = mm.Tessellation(
    ... buildings_df, 'uID', enclosures=enclosures
    ... )
    >>> encl_tess.tessellation.head()
         uID                                           geometry  eID
    0  109.0  POLYGON ((1603369.789 6464340.661, 1603368.754...    0
    1  110.0  POLYGON ((1603368.754 6464340.097, 1603369.789...    0
    2  111.0  POLYGON ((1603458.666 6464332.614, 1603458.332...    0
    3  112.0  POLYGON ((1603462.235 6464285.609, 1603454.795...    0
    4  113.0  POLYGON ((1603524.561 6464388.609, 1603532.241...    0

    """

    def __init__(
        self,
        gdf,
        unique_id,
        limit=None,
        shrink=0.4,
        segment=0.5,
        verbose=True,
        enclosures=None,
        enclosure_id="eID",
        threshold=0.05,
        use_dask=True,
        n_chunks=None,
        **kwargs,
    ):
        self.gdf = gdf
        self.id = gdf[unique_id]
        self.limit = limit
        self.shrink = shrink
        self.segment = segment
        self.enclosure_id = enclosure_id

        if limit is not None and enclosures is not None:
            raise ValueError(
                "Both `limit` and `enclosures` cannot be passed together. "
                "Pass `limit` for morphological tessellation or `enclosures` "
                "for enclosed tessellation."
            )
        if enclosures is not None:
            self.tessellation = self._enclosed_tessellation(
                gdf, enclosures, unique_id, enclosure_id, threshold, use_dask, n_chunks,
            )
        else:
            self.tessellation = self._morphological_tessellation(
                gdf, unique_id, limit, shrink, segment, verbose
            )

    def _morphological_tessellation(
        self, gdf, unique_id, limit, shrink, segment, verbose, check=True
    ):
        objects = gdf.copy()

        if isinstance(limit, (gpd.GeoSeries, gpd.GeoDataFrame)):
            limit = limit.unary_union
        if isinstance(limit, BaseGeometry):
            limit = pygeos.from_shapely(limit)

        bounds = pygeos.bounds(limit)
        centre_x = (bounds[0] + bounds[2]) / 2
        centre_y = (bounds[1] + bounds[3]) / 2
        objects["geometry"] = objects["geometry"].translate(
            xoff=-centre_x, yoff=-centre_y
        )

        if shrink != 0:
            print("Inward offset...") if verbose else None
            mask = objects.type.isin(["Polygon", "MultiPolygon"])
            objects.loc[mask, "geometry"] = objects[mask].buffer(
                -shrink, cap_style=2, join_style=2
            )

        objects = objects.reset_index(drop=True).explode()
        objects = objects.set_index(unique_id)

        print("Generating input point array...") if verbose else None
        points, ids = self._dense_point_array(
            objects.geometry.values.data, distance=segment, index=objects.index
        )

        # add convex hull buffered large distance to eliminate infinity issues
        series = gpd.GeoSeries(limit, crs=gdf.crs).translate(
            xoff=-centre_x, yoff=-centre_y
        )
        width = bounds[2] - bounds[0]
        leng = bounds[3] - bounds[1]
        hull = series.geometry[[0]].buffer(2 * width if width > leng else 2 * leng)
        # pygeos bug fix
        if (hull.type == "MultiPolygon").any():
            hull = hull.explode()
        hull_p, hull_ix = self._dense_point_array(
            hull.values.data, distance=pygeos.length(limit) / 100, index=hull.index
        )
        points = np.append(points, hull_p, axis=0)
        ids = ids + ([-1] * len(hull_ix))

        print("Generating Voronoi diagram...") if verbose else None
        voronoi_diagram = Voronoi(np.array(points))

        print("Generating GeoDataFrame...") if verbose else None
        regions_gdf = self._regions(voronoi_diagram, unique_id, ids, crs=gdf.crs)

        print("Dissolving Voronoi polygons...") if verbose else None
        morphological_tessellation = regions_gdf[[unique_id, "geometry"]].dissolve(
            by=unique_id, as_index=False
        )

        morphological_tessellation = gpd.clip(morphological_tessellation, series)

        morphological_tessellation["geometry"] = morphological_tessellation[
            "geometry"
        ].translate(xoff=centre_x, yoff=centre_y)

        if check:
            self._check_result(morphological_tessellation, gdf, unique_id=unique_id)

        return morphological_tessellation

    def _dense_point_array(self, geoms, distance, index):
        """
        geoms - array of pygeos lines
        """
        # interpolate lines to represent them as points for Voronoi
        points = np.empty((0, 2))
        ids = []

        if pygeos.get_type_id(geoms[0]) not in [1, 2, 5]:
            lines = pygeos.boundary(geoms)
        else:
            lines = geoms
        lengths = pygeos.length(lines)
        for ix, line, length in zip(index, lines, lengths):
            if length > distance:  # some polygons might have collapsed
                pts = pygeos.line_interpolate_point(
                    line,
                    np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance)),
                )  # .1 offset to keep a gap between two segments
                points = np.append(points, pygeos.get_coordinates(pts), axis=0)
                ids += [ix] * len(pts)

        return points, ids

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners

    def _regions(self, voronoi_diagram, unique_id, ids, crs):
        """
        Generate GeoDataFrame of Voronoi regions from scipy.spatial.Voronoi.
        """
        vertices = pd.Series(voronoi_diagram.regions).take(voronoi_diagram.point_region)
        polygons = []
        for region in vertices:
            if -1 not in region:
                polygons.append(pygeos.polygons(voronoi_diagram.vertices[region]))
            else:
                polygons.append(None)

        regions_gdf = gpd.GeoDataFrame(
            {unique_id: ids}, geometry=polygons, crs=crs
        ).dropna()
        regions_gdf = regions_gdf.loc[
            regions_gdf[unique_id] != -1
        ]  # delete hull-based cells

        return regions_gdf

    def _check_result(self, tesselation, orig_gdf, unique_id):
        """
        Check whether result of tessellation matches buildings and contains only Polygons.
        """
        # check against input layer
        ids_original = list(orig_gdf[unique_id])
        ids_generated = list(tesselation[unique_id])
        if len(ids_original) != len(ids_generated):
            import warnings

            self.collapsed = set(ids_original).difference(ids_generated)
            warnings.warn(
                "Tessellation does not fully match buildings. {len} element(s) collapsed "
                "during generation - unique_id: {i}".format(
                    len=len(self.collapsed), i=self.collapsed
                )
            )

        # check MultiPolygons - usually caused by error in input geometry
        self.multipolygons = tesselation[tesselation.geometry.type == "MultiPolygon"][
            unique_id
        ]
        if len(self.multipolygons) > 0:
            import warnings

            warnings.warn(
                "Tessellation contains MultiPolygon elements. Initial objects should be edited. "
                "unique_id of affected elements: {}".format(list(self.multipolygons))
            )

    def _enclosed_tessellation(
        self,
        buildings,
        enclosures,
        unique_id,
        enclosure_id="eID",
        threshold=0.05,
        use_dask=True,
        n_chunks=None,
        **kwargs,
    ):
        """Enclosed tessellation
        Generate enclosed tessellation based on barriers defining enclosures and buildings
        footprints.

        Parameters
        ----------
        buildings : GeoDataFrame
            GeoDataFrame containing building footprints. Expects (Multi)Polygon geometry.
        enclosures : GeoDataFrame
            Enclosures geometry. Can  be generated using :func:`momepy.enclosures`.
        unique_id : str
            name of the column with unique id of buildings gdf
        threshold : float (default 0.05)
            The minimum threshold for a building to be considered within an enclosure.
            Threshold is a ratio of building area which needs to be within an enclosure to
            inlude it in the tessellation of that enclosure. Resolves sliver geometry
            issues.
        use_dask : bool (default True)
            Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        n_chunks : None
            Number of chunks to be used in parallelization. Ideal is one chunk per thread.
            Applies only if ``enclosures`` are passed. Defualt automatically uses
            n == dask.system.cpu_count.
        **kwargs
            Keyword arguments passed to Tessellation algorithm (as ``shrink``
            or ``segment``).

        Returns
        -------
        tessellation : GeoDataFrame
            gdf contains three columns:
                geometry,
                unique_id matching with parental building,
                enclosure_id matching with enclosure integer index

        Examples
        --------
        >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
        >>> enclosed_tess = mm.enclosed_tessellation(buildings, enclosures)

        """
        enclosures = enclosures.reset_index(drop=True)

        # determine which polygons should be split
        inp, res = buildings.sindex.query_bulk(
            enclosures.geometry, predicate="intersects"
        )
        unique, counts = np.unique(inp, return_counts=True)
        splits = unique[counts > 1]
        single = unique[counts == 1]

        if use_dask:
            try:
                import dask.dataframe as dd
                from dask.system import cpu_count
            except ImportError:
                use_dask = False

                import warnings

                warnings.warn(
                    "dask.dataframe could not be imported. Setting `use_dask=False`."
                )

        if use_dask:
            if n_chunks is None:
                n_chunks = cpu_count() - 1 if cpu_count() > 1 else 1
            # initialize dask.series
            ds = dd.from_array(splits, chunksize=len(splits) // n_chunks)
            # generate enclosed tessellation using dask
            new = (
                ds.apply(
                    self._tess,
                    meta=(None, "object"),
                    args=(enclosures, buildings, inp, res, threshold, unique_id),
                )
                .compute()
                .to_list()
            )

        else:
            new = [
                self._tess(
                    i,
                    enclosures,
                    buildings,
                    inp,
                    res,
                    threshold=threshold,
                    unique_id=unique_id,
                    **kwargs,
                )
                for i in splits
            ]

        # finalise the result
        clean_blocks = enclosures.drop(splits)
        clean_blocks.loc[single, "uID"] = clean_blocks.loc[single][enclosure_id].apply(
            lambda ix: buildings.iloc[res[inp == ix][0]][unique_id]
        )
        tessellation = pd.concat(new)

        return tessellation.append(clean_blocks).reset_index(drop=True)

    def _tess(
        self,
        ix,
        enclosure,
        buildings,
        query_inp,
        query_res,
        threshold,
        unique_id,
        **kwargs,
    ):
        poly = enclosure.geometry.values.data[ix]
        blg = buildings.iloc[query_res[query_inp == ix]]
        within = blg[
            pygeos.area(pygeos.intersection(blg.geometry.values.data, poly))
            > (pygeos.area(blg.geometry.values.data) * threshold)
        ]
        if len(within) > 1:
            tess = self._morphological_tessellation(
                within,
                unique_id,
                poly,
                shrink=self.shrink,
                segment=self.segment,
                verbose=False,
                check=False,
            )
            tess[self.enclosure_id] = enclosure[self.enclosure_id].iloc[ix]
            return tess
        return gpd.GeoDataFrame(
            {self.enclosure_id: enclosure[self.enclosure_id].iloc[ix], unique_id: None},
            geometry=[poly],
            index=[0],
        )


class Blocks:
    """
    Generate blocks based on buildings, tesselation and street network.

    Dissolves tessellation cells based on street-network based polygons.
    Links resulting id to ``buildings`` and ``tesselation`` as attributes.

    Parameters
    ----------
    tessellation : GeoDataFrame
        GeoDataFrame containing morphological tessellation
    edges : GeoDataFrame
        GeoDataFrame containing street network
    buildings : GeoDataFrame
        GeoDataFrame containing buildings
    id_name : str
        name of the unique blocks id column to be generated
    unique_id : str
        name of the column with unique id. If there is none, it could be generated by :func:`momepy.unique_id`.
        This should be the same for cells and buildings, id's should match.

    Attributes
    ----------
    blocks : GeoDataFrame
        GeoDataFrame containing generated blocks
    buildings_id : Series
        Series derived from buildings with block ID
    tessellation_id : Series
        Series derived from morphological tessellation with block ID
    tessellation : GeoDataFrame
        GeoDataFrame containing original tessellation
    edges : GeoDataFrame
        GeoDataFrame containing original edges
    buildings : GeoDataFrame
        GeoDataFrame containing original buildings
    id_name : str
        name of the unique blocks id column
    unique_id : str
        name of the column with unique id

    Examples
    --------
    >>> blocks_generate = mm.Blocks(tessellation_df, streets_df, buildings_df, 'bID', 'uID')
    Buffering streets...
    Generating spatial index...
    Difference...
    Defining adjacency...
    Defining street-based blocks...
    Defining block ID...
    Generating centroids...
    Spatial join...
    Attribute join (tesselation)...
    Generating blocks...
    Multipart to singlepart...
    Attribute join (buildings)...
    Attribute join (tesselation)...
    >>> blocks_generate.blocks.head()
        bID	geometry
    0	1.0	POLYGON ((1603560.078648818 6464202.366899694,...
    1	2.0	POLYGON ((1603457.225976106 6464299.454696888,...
    2	3.0	POLYGON ((1603056.595487018 6464093.903488506,...
    3	4.0	POLYGON ((1603260.943782872 6464141.327631323,...
    4	5.0	POLYGON ((1603183.399594798 6463966.109982309,...

    """

    def __init__(self, tessellation, edges, buildings, id_name, unique_id, **kwargs):
        self.tessellation = tessellation
        self.edges = edges
        self.buildings = buildings
        self.id_name = id_name
        self.unique_id = unique_id

        if id_name in buildings.columns:
            raise ValueError(
                "'{}' column cannot be in the buildings GeoDataFrame".format(id_name)
            )

        cut = gpd.overlay(
            tessellation,
            gpd.GeoDataFrame(geometry=edges.buffer(0.001)),
            how="difference",
        ).explode()

        W = libpysal.weights.Queen.from_dataframe(cut, silence_warnings=True)
        cut["component"] = W.component_labels
        buildings_c = buildings.copy()
        buildings_c["geometry"] = buildings_c.representative_point()  # make points
        centroids_tempID = gpd.sjoin(
            buildings_c, cut[["geometry", "component"]], how="left", op="intersects"
        )
        cells_copy = tessellation[[unique_id, "geometry"]].merge(
            centroids_tempID[[unique_id, "component"]], on=unique_id, how="left"
        )
        blocks = cells_copy.dissolve(by="component").explode().reset_index(drop=True)
        blocks[id_name] = range(len(blocks))
        blocks["geometry"] = gpd.GeoSeries(
            pygeos.polygons(blocks.exterior.values.data), crs=blocks.crs
        )
        blocks = blocks[[id_name, "geometry"]]

        centroids_w_bl_ID2 = gpd.sjoin(buildings_c, blocks, how="left", op="intersects")
        bl_ID_to_uID = centroids_w_bl_ID2[[unique_id, id_name]]

        buildings_m = buildings[[unique_id]].merge(
            bl_ID_to_uID, on=unique_id, how="left"
        )
        self.buildings_id = buildings_m[id_name]

        cells_m = tessellation[[unique_id]].merge(
            bl_ID_to_uID, on=unique_id, how="left"
        )
        self.tessellation_id = cells_m[id_name]

        self.blocks = blocks


def get_network_id(left, right, network_id, min_size=100, verbose=True):
    """
    Snap each element (preferably building) to the closest street network segment,
    saves its id.

    Adds network ID to elements.

    Parameters
    ----------
    left : GeoDataFrame
        GeoDataFrame containing objects to snap
    right : GeoDataFrame
        GeoDataFrame containing street network with unique network ID.
        If there is none, it could be generated by :func:`momepy.unique_id`.
    network_id : str, list, np.array, pd.Series (default None)
        the name of the streets dataframe column, ``np.array``, or ``pd.Series``
        with network unique id.
    min_size : int (default 100)
        min_size should be a vaule such that if you build a box centered in each
        building centroid with edges of size ``2*min_size``, you know a priori that at
        least one
        segment is intersected with the box.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    elements_nID : Series
        Series containing network ID for elements

    Examples
    --------
    >>> buildings_df['nID'] = momepy.get_network_id(buildings_df, streets_df, 'nID')
    Generating centroids...
    Generating rtree...
    Snapping: 100%|██████████| 144/144 [00:00<00:00, 2718.98it/s]
    >>> buildings_df['nID'][0]
    1

    See also
    --------
    momepy.get_network_ratio
    momepy.get_node_id
    """
    INFTY = 1000000000000
    left = left.copy()
    right = right.copy()

    if not isinstance(network_id, str):
        right["mm_nid"] = network_id
        network_id = "mm_nid"

    print("Generating centroids...") if verbose else None
    buildings_c = left.copy()

    buildings_c["geometry"] = buildings_c.centroid  # make centroids

    print("Generating rtree...") if verbose else None
    idx = right.sindex

    # TODO: use sjoin nearest once done
    result = []
    for p in tqdm(
        buildings_c.geometry,
        total=buildings_c.shape[0],
        desc="Snapping",
        disable=not verbose,
    ):
        pbox = (p.x - min_size, p.y - min_size, p.x + min_size, p.y + min_size)
        hits = list(idx.intersection(pbox))
        d = INFTY
        nid = None
        for h in hits:
            new_d = p.distance(right.geometry.iloc[h])
            if d >= new_d:
                d = new_d
                nid = right[network_id].iloc[h]
        if nid is None:
            result.append(np.nan)
        else:
            result.append(nid)

    series = pd.Series(result)

    if series.isnull().any():
        import warnings

        warnings.warn(
            "Some objects were not attached to the network. "
            "Set larger min_size. {} affected elements".format(sum(series.isnull()))
        )
    return series


def get_node_id(
    objects,
    nodes,
    edges,
    node_id,
    edge_id=None,
    edge_keys=None,
    edge_values=None,
    verbose=True,
):
    """
    Snap each building to closest street network node on the closest network edge.

    Adds node ID to objects (preferably buildings). Gets ID of edge
    (:func:`momepy.get_network_id` or :func:`get_network_ratio`)
    , and determines which of its end points is closer to building centroid.

    Pass either ``edge_id`` with a single value or ``edge_keys`` and ``edge_values``
    with ratios.

    Parameters
    ----------
    objects : GeoDataFrame
        GeoDataFrame containing objects to snap
    nodes : GeoDataFrame
        GeoDataFrame containing street nodes with unique node ID.
        If there is none, it could be generated by :func:`momepy.unique_id`.
    edges : GeoDataFrame
        GeoDataFrame containing street edges with unique edge ID and IDs of start
        and end points of each segment. Start and endpoints are default
        outcome of :func:`momepy.nx_to_gdf`.
    node_id : str, list, np.array, pd.Series
        the name of the nodes dataframe column, ``np.array``, or ``pd.Series``
        with unique id
    edge_id : str (default None)
        the name of the objects dataframe column
        with unique edge id (like an outcome of :func:`momepy.get_network_id`)
    edge_keys : str (default None)
        name the name of the objects dataframe column with edgeID_keys
        (like an outcome of :func:`momepy.get_network_ratio`)
    edge_values : str (default None)
        name the name of the objects dataframe column with edgeID_values
        (like an outcome of :func:`momepy.get_network_ratio`)
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Returns
    -------
    node_ids : Series
        Series containing node ID for objects

    """
    nodes = nodes.set_index(node_id)

    if not isinstance(node_id, str):
        nodes["mm_noid"] = node_id
        node_id = "mm_noid"

    results_list = []
    if edge_id is not None:
        edges = edges.set_index(edge_id)
        centroids = objects.centroid
        for eid, centroid in tqdm(
            zip(objects[edge_id], centroids),
            total=objects.shape[0],
            disable=not verbose,
        ):
            if np.isnan(eid):
                results_list.append(np.nan)
            else:
                edge = edges.loc[eid]
                startID = edge.node_start
                start = nodes.loc[startID].geometry
                sd = centroid.distance(start)
                endID = edge.node_end
                end = nodes.loc[endID].geometry
                ed = centroid.distance(end)
                if sd > ed:
                    results_list.append(endID)
                else:
                    results_list.append(startID)

    elif edge_keys is not None and edge_values is not None:
        for edge_i, edge_r, geom in tqdm(
            zip(objects[edge_keys], objects[edge_values], objects.geometry),
            total=objects.shape[0],
            disable=not verbose,
        ):
            edge = edges.iloc[edge_i[edge_r.index(max(edge_r))]]
            startID = edge.node_start
            start = nodes.loc[startID].geometry
            sd = geom.distance(start)
            endID = edge.node_end
            end = nodes.loc[endID].geometry
            ed = geom.distance(end)
            if sd > ed:
                results_list.append(endID)
            else:
                results_list.append(startID)

    series = pd.Series(results_list, index=objects.index)
    return series


def get_network_ratio(df, edges, initial_buffer=500):
    """
    Link polygons to network edges based on the proportion of overlap (if a cell
    intersects more than one edge)

    Useful if you need to link enclosed tessellation to street network. Ratios can
    be used as weights when linking network-based values to cells. For a purely
    distance-based link use :func:`momepy.get_network_id`.

    Links are based on the integer position of edge (``iloc``).

    Parameters
    ----------

    df : GeoDataFrame
        GeoDataFrame containing objects to snap (typically enclosed tessellation)
    edges : GeoDataFrame
        GeoDataFrame containing street network
    initial_buffer : float
        Initial buffer used to link non-intersecting cells.

    Returns
    -------

    DataFrame

    See also
    --------
    momepy.get_network_id
    momepy.get_node_id

    Examples
    --------
    >>> links = mm.get_network_ratio(enclosed_tessellation, streets)
    >>> links.head()
      edgeID_keys                              edgeID_values
    0        [34]                                      [1.0]
    1     [0, 34]  [0.38508998545027145, 0.6149100145497285]
    2        [32]                                        [1]
    3         [0]                                      [1.0]
    4        [26]                                        [1]

    """

    # intersection-based join
    buff = edges.buffer(0.01)  # to avoid floating point error
    inp, res = buff.sindex.query_bulk(df.geometry, predicate="intersects")
    intersections = (
        df.iloc[inp]
        .reset_index(drop=True)
        .intersection(buff.iloc[res].reset_index(drop=True))
    )
    mask = intersections.area > 0.0001
    intersections = intersections[mask]
    inp = inp[mask]
    lengths = intersections.area
    grouped = lengths.groupby(inp)
    totals = grouped.sum()
    ints_vect = []
    for name, group in grouped:
        ratios = group / totals.loc[name]
        ints_vect.append({res[item[0]]: item[1] for item in ratios.iteritems()})

    edge_dicts = pd.Series(ints_vect, index=totals.index)

    # nearest neighbor join
    nans = df.index.difference(edge_dicts.index)
    buffered = df.iloc[nans].buffer(initial_buffer)
    additional = []
    for orig, geom in zip(df.iloc[nans].geometry, buffered.geometry):
        query = edges.sindex.query(geom, predicate="intersects")
        b = initial_buffer
        while query.size == 0:
            query = edges.sindex.query(geom.buffer(b), predicate="intersects")
            b += initial_buffer
        additional.append({edges.iloc[query].distance(orig).idxmin(): 1})

    additional = pd.Series(additional, index=nans)
    ratios = pd.concat([edge_dicts, additional]).sort_index()
    result = pd.DataFrame()
    result["edgeID_keys"] = ratios.apply(lambda d: list(d.keys()))
    result["edgeID_values"] = ratios.apply(lambda d: list(d.values()))
    result.index = df.index
    return result


def _split_lines(polygon, distance):
    """Split polygon into GeoSeries of lines no longer than `distance`."""
    list_points = []
    current_dist = distance  # set the current distance to place the point

    # TODO: use pygeos interpolate
    boundary = polygon.boundary  # make shapely MultiLineString object
    if boundary.type == "LineString":
        line_length = boundary.length  # get the total length of the line
        while (
            current_dist < line_length
        ):  # while the current cumulative distance is less than the total length of the line
            list_points.append(
                boundary.interpolate(current_dist)
            )  # use interpolate and increase the current distance
            current_dist += distance
    elif boundary.type == "MultiLineString":
        for ls in boundary:
            line_length = ls.length  # get the total length of the line
            while (
                current_dist < line_length
            ):  # while the current cumulative distance is less than the total length of the line
                list_points.append(
                    ls.interpolate(current_dist)
                )  # use interpolate and increase the current distance
                current_dist += distance

    cutted = shapely.ops.split(
        boundary, shapely.geometry.MultiPoint(list_points).buffer(0.001)
    )
    return cutted


def enclosures(
    primary_barriers, limit=None, additional_barriers=None, enclosure_id="eID"
):
    """
    Generate enclosures based on passed barriers.

    Enclosures are areas enclosed from all sides by at least one type of
    a barrier. Barriers are typically roads, railways, natural features
    like rivers and other water bodies or coastline. Enclosures are a
    result of polygonization of the  ``primary_barrier`` and ``limit`` and its
    subdivision based on additional_barriers.

    Parameters
    ----------
    primary_barriers : GeoDataFrame, GeoSeries
        GeoDataFrame or GeoSeries containing primary barriers.
        (Multi)LineString geometry is expected.
    limit : GeoDataFrame, GeoSeries (default None)
        GeoDataFrame or GeoSeries containing external limit of enclosures,
        i.e. the area which gets partitioned. If None is passed,
        the internal area of ``primary_barriers`` will be used.
    additional_barriers : GeoDataFrame
        GeoDataFrame or GeoSeries containing additional barriers.
        (Multi)LineString geometry is expected.
    enclosure_id : str (default 'eID')
        name of the enclosure_id (to be created).

    Returns
    -------
    enclosures : GeoDataFrame
       GeoDataFrame containing enclosure geometries and enclosure_id

    Examples
    --------
    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])

    """
    if limit is not None:
        if limit.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            limit = limit.boundary
        barriers = pd.concat([primary_barriers.geometry, limit.geometry])
    else:
        barriers = primary_barriers
    unioned = barriers.unary_union
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=primary_barriers.crs)

    if additional_barriers is not None:
        if not isinstance(additional_barriers, list):
            raise TypeError(
                "`additional_barriers` expects a list of GeoDataFrames or GeoSeries."
                f"Got {type(additional_barriers)}."
            )
        additional = pd.concat([gdf.geometry for gdf in additional_barriers])

        inp, res = enclosures.sindex.query_bulk(
            additional.geometry, predicate="intersects"
        )
        unique = np.unique(res)

        new = []

        for i in unique:
            poly = enclosures.values.data[i]  # get enclosure polygon
            crossing = inp[res == i]  # get relevant additional barriers
            buf = pygeos.buffer(poly, 0.01)  # to avoid floating point errors
            crossing_ins = pygeos.intersection(
                buf, additional.values.data[crossing]
            )  # keeping only parts of additional barriers within polygon
            union = pygeos.union_all(
                np.append(crossing_ins, pygeos.boundary(poly))
            )  # union
            polygons = np.array(
                list(polygonize(_pygeos_to_shapely(union)))
            )  # polygonize
            within = pygeos.covered_by(
                pygeos.from_shapely(polygons), buf
            )  # keep only those within original polygon
            new += list(polygons[within])

        final_enclosures = (
            gpd.GeoSeries(enclosures)
            .drop(unique)
            .append(gpd.GeoSeries(new))
            .reset_index(drop=True)
        ).set_crs(primary_barriers.crs)

        return gpd.GeoDataFrame(
            {enclosure_id: range(len(final_enclosures))}, geometry=final_enclosures
        )

    return gpd.GeoDataFrame({enclosure_id: range(len(enclosures))}, geometry=enclosures)
