#!/usr/bin/env python

# elements.py
# generating derived elements (street edge, block)
import warnings

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import shapely
from packaging.version import Version
from scipy.spatial import Voronoi
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize
from tqdm.auto import tqdm

__all__ = [
    "buffered_limit",
    "Tessellation",
    "Blocks",
    "get_network_id",
    "get_node_id",
    "enclosures",
    "get_network_ratio",
]

GPD_10 = Version(gpd.__version__) >= Version("0.10")


def buffered_limit(gdf, buffer=100):
    """
    Define limit for :class:`momepy.Tessellation` as a buffer around buildings.

    See :cite:`fleischmann2020` for details.

    Parameters
    ----------
    gdf : GeoDataFrame
        A GeoDataFrame containing building footprints.
    buffer : float
        A buffer around buildings limiting the extend of tessellation.

    Returns
    -------
    MultiPolygon
        A MultiPolygon or Polygon defining the study area.

    Examples
    --------
    >>> limit = mm.buffered_limit(buildings_df)
    >>> type(limit)
    shapely.geometry.polygon.Polygon
    """
    return gdf.buffer(buffer).unary_union


class Tessellation:
    """
    Generates tessellation. Three versions of tessellation can be created:

        1. Morphological tessellation around given buildings
            ``gdf`` within set ``limit``.
        2. Proximity bands around given street network ``gdf``
            within set ``limit``.
        3. Enclosed tessellation based on given buildings
            ``gdf`` within ``enclosures``.

    Pass either ``limit`` to create morphological tessellation or proximity bands or
    ``enclosures`` to create enclosed tessellation.

    See :cite:`fleischmann2020` for details of implementation of morphological
    tessellation and :cite:`araldi2019` for proximity bands.

    Tessellation requires data of relatively high level of precision
    and there are three particular patterns causing issues:

        1. Features will collapse into empty polygon - these
            do not have tessellation cell in the end.
        2. Features will split into MultiPolygons - in some cases,
            features with narrow links between parts split into two
            during 'shrinking'. In most cases that is not an issue
            and the resulting tessellation is correct anyway, but
            sometimes this results in a cell being a MultiPolygon,
            which is not correct.
        3. Overlapping features - features which overlap even
            after 'shrinking' cause invalid tessellation geometry.

    All three types can be tested prior :class:`momepy.Tessellation` using
    :class:`momepy.CheckTessellationInput`.

    Parameters
    ----------
    gdf : GeoDataFrame
       A GeoDataFrame containing building footprints or street network.
    unique_id : str
        The name of the column with the unique ID.
    limit : MultiPolygon or Polygon (default None)
        MultiPolygon or Polygon defining the study area limiting morphological
        tessellation or proximity bands (otherwise it could go to infinity).
    shrink : float (default 0.4)
        The distance for negative buffer to generate space between adjacent polygons
        (if geometry type of gdf is (Multi)Polygon).
    segment : float (default 0.5)
        The maximum distance between points after discretization.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.
    enclosures : GeoDataFrame (default None)
        The enclosures geometry, which can be generated
        using :func:`momepy.enclosures`.
    enclosure_id : str (default 'eID')
        The name of the ``enclosure_id`` containing unique identifer for each row in
        ``enclosures``. Applies only if ``enclosures`` are passed.
    threshold : float (default 0.05)
        The minimum threshold for a building to be considered within an enclosure.
        Threshold is a ratio of building area which needs to be within an enclosure to
        inlude it in the tessellation of that enclosure. Resolves sliver geometry
        issues. Applies only if ``enclosures`` are passed.
    use_dask : bool (default True)
        Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        Applies only if ``enclosures`` are passed.
    n_chunks : None
        The number of chunks to be used in parallelization. Ideal is one chunk per
        thread. Applies only if ``enclosures`` are passed. Default automatically
        uses ``n == dask.system.cpu_count``.

    Attributes
    ----------
    tessellation : GeoDataFrame
        A GeoDataFrame containing resulting tessellation.
        For enclosed tessellation, gdf contains three columns:

            - ``geometry``,
            - ``unique_id`` matching with parental building,
            - ``enclosure_id`` matching with enclosure integer index

    gdf : GeoDataFrame
        The original GeoDataFrame.
    id : Series
        A Series containing used unique ID.
    limit : MultiPolygon or Polygon
        MultiPolygon or Polygon defining the study area limiting morphological
        tessellation or proximity bands.
    shrink : float
        The distance for negative buffer to generate space between adjacent polygons.
    segment : float
        The maximum distance between points after discretization.
    collapsed : list
        A list of ``unique_id``s of collapsed features (if there are any).
        Applies only if ``limit`` is passed.
    multipolygons : list
        A list of ``unique_id``s of features causing MultiPolygons (if there are any).
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
    ):
        self.gdf = gdf
        self.id = gdf[unique_id]
        self.limit = limit
        self.shrink = shrink
        self.segment = segment
        self.enclosure_id = enclosure_id

        if gdf.crs and gdf.crs.is_geographic:
            raise ValueError(
                "Geometry is in a geographic CRS. "
                "Use 'GeoDataFrame.to_crs()' to re-project geometries to a "
                "projected CRS before using Tessellation.",
            )

        if limit is not None and enclosures is not None:
            raise ValueError(
                "Both `limit` and `enclosures` cannot be passed together. "
                "Pass `limit` for morphological tessellation or `enclosures` "
                "for enclosed tessellation."
            )

        gdf = gdf.copy()

        if enclosures is not None:
            enclosures = enclosures.copy()

            bounds = enclosures.total_bounds
            centre_x = (bounds[0] + bounds[2]) / 2
            centre_y = (bounds[1] + bounds[3]) / 2

            gdf.geometry = gdf.geometry.translate(xoff=-centre_x, yoff=-centre_y)
            enclosures.geometry = enclosures.geometry.translate(
                xoff=-centre_x, yoff=-centre_y
            )

            self.tessellation = self._enclosed_tessellation(
                gdf,
                enclosures,
                unique_id,
                threshold,
                use_dask,
                n_chunks,
            )
        else:
            if isinstance(limit, (gpd.GeoSeries, gpd.GeoDataFrame)):
                limit = limit.unary_union

            bounds = shapely.bounds(limit)
            centre_x = (bounds[0] + bounds[2]) / 2
            centre_y = (bounds[1] + bounds[3]) / 2
            gdf.geometry = gdf.geometry.translate(xoff=-centre_x, yoff=-centre_y)

            # add convex hull buffered large distance to eliminate infinity issues
            limit = (
                gpd.GeoSeries(limit, crs=gdf.crs)
                .translate(xoff=-centre_x, yoff=-centre_y)
                .array[0]
            )

            self.tessellation = self._morphological_tessellation(
                gdf, unique_id, limit, shrink, segment, verbose
            )

        self.tessellation["geometry"] = self.tessellation["geometry"].translate(
            xoff=centre_x, yoff=centre_y
        )

    def _morphological_tessellation(
        self, gdf, unique_id, limit, shrink, segment, verbose, check=True
    ):
        objects = gdf

        if shrink != 0:
            print("Inward offset...") if verbose else None
            mask = objects.geom_type.isin(["Polygon", "MultiPolygon"])
            objects.loc[mask, objects.geometry.name] = objects[mask].buffer(
                -shrink, cap_style=2, join_style=2
            )
        if GPD_10:
            objects = objects.reset_index(drop=True).explode(ignore_index=True)
        else:
            objects = objects.reset_index(drop=True).explode().reset_index(drop=True)
        objects = objects.set_index(unique_id)

        print("Generating input point array...") if verbose else None
        points, ids = self._dense_point_array(
            objects.geometry.array, distance=segment, index=objects.index
        )

        hull = shapely.convex_hull(limit)
        bounds = shapely.bounds(hull)
        width = bounds[2] - bounds[0]
        leng = bounds[3] - bounds[1]
        hull = shapely.buffer(hull, 2 * width if width > leng else 2 * leng)

        hull_p, hull_ix = self._dense_point_array(
            [hull], distance=shapely.length(hull) / 100, index=[0]
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

        morphological_tessellation = gpd.clip(
            morphological_tessellation, gpd.GeoSeries(limit, crs=gdf.crs)
        )

        if check:
            self._check_result(morphological_tessellation, gdf, unique_id=unique_id)

        return morphological_tessellation

    def _dense_point_array(self, geoms, distance, index):
        """
        geoms : array of shapely lines
        """
        # interpolate lines to represent them as points for Voronoi
        points = []
        ids = []

        if shapely.get_type_id(geoms[0]) not in [1, 2, 5]:
            lines = shapely.boundary(geoms)
        else:
            lines = geoms
        lengths = shapely.length(lines)
        for ix, line, length in zip(index, lines, lengths):
            if length > distance:  # some polygons might have collapsed
                pts = shapely.line_interpolate_point(
                    line,
                    np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance)),
                )  # .1 offset to keep a gap between two segments
                points.append(shapely.get_coordinates(pts))
                ids += [ix] * len(pts)

        points = np.vstack(points)

        return points, ids

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners

    def _regions(self, voronoi_diagram, unique_id, ids, crs):
        """Generate GeoDataFrame of Voronoi regions from scipy.spatial.Voronoi."""
        vertices = pd.Series(voronoi_diagram.regions).take(voronoi_diagram.point_region)
        polygons = []
        for region in vertices:
            if -1 not in region:
                polygons.append(shapely.polygons(voronoi_diagram.vertices[region]))
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
        """Check whether result matches buildings and contains only Polygons."""
        # check against input layer
        ids_original = list(orig_gdf[unique_id])
        ids_generated = list(tesselation[unique_id])
        if len(ids_original) != len(ids_generated):
            self.collapsed = set(ids_original).difference(ids_generated)
            warnings.warn(
                message=(
                    "Tessellation does not fully match buildings. "
                    f"{len(self.collapsed)} element(s) collapsed "
                    f"during generation - unique_id: {self.collapsed}."
                ),
                category=UserWarning,
                stacklevel=4,
            )

        # check MultiPolygons - usually caused by error in input geometry
        self.multipolygons = tesselation[
            tesselation.geometry.geom_type == "MultiPolygon"
        ][unique_id]
        if len(self.multipolygons) > 0:
            warnings.warn(
                message=(
                    "Tessellation contains MultiPolygon elements. Initial "
                    "objects should  be edited. `unique_id` of affected "
                    f"elements: {list(self.multipolygons)}."
                ),
                category=UserWarning,
                stacklevel=4,
            )

    def _enclosed_tessellation(
        self,
        buildings,
        enclosures,
        unique_id,
        threshold=0.05,
        use_dask=True,
        n_chunks=None,
        **kwargs,
    ):
        """
        Generate enclosed tessellation based on barriers
        defining enclosures and building footprints.

        Parameters
        ----------
        buildings : GeoDataFrame
            A GeoDataFrame containing building footprints.
            Expects (Multi)Polygon geometry.
        enclosures : GeoDataFrame
            Enclosures geometry. Can  be generated using :func:`momepy.enclosures`.
        unique_id : str
            The name of the column with the unique ID of ``buildings`` gdf.
        threshold : float (default 0.05)
            The minimum threshold for a building to be considered within an enclosure.
            Threshold is a ratio of building area which needs to be within an enclosure
            to inlude it in the tessellation of that enclosure.
            Resolves sliver geometry issues.
        use_dask : bool (default True)
            Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        n_chunks : None
            The number of chunks to be used in parallelization. Ideal is one chunk per
            thread. Applies only if ``enclosures`` are passed. Default automatically
            uses ``n == dask.system.cpu_count``.
        **kwargs : dict
            Keyword arguments passed to Tessellation algorithm
            (such as ``'shrink'`` or ``'segment'``).

        Returns
        -------
        tessellation : GeoDataFrame
            A GeoDataFrame containing three columns:

                - ``geometry``,
                - ``unique_id`` matching with parent building,
                - ``enclosure_id`` matching with enclosure integer index

        Examples
        --------
        >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
        >>> enclosed_tess = mm.enclosed_tessellation(buildings, enclosures)
        """
        enclosures = enclosures.reset_index(drop=True)
        enclosures["position"] = range(len(enclosures))

        # determine which polygons should be split
        inp, res = buildings.sindex.query_bulk(
            enclosures.geometry, predicate="intersects"
        )
        unique, counts = np.unique(inp, return_counts=True)
        splits = unique[counts > 1]
        single = unique[counts == 1]

        if use_dask:
            try:
                import dask.bag as db
                from dask.system import cpu_count
            except ImportError:
                use_dask = False

                warnings.warn(
                    message=(
                        "dask.dataframe could not be imported. "
                        f"Setting `use_dask={use_dask}`."
                    ),
                    category=UserWarning,
                    stacklevel=3,
                )

        if use_dask:
            if n_chunks is None:
                n_chunks = cpu_count() - 1 if cpu_count() > 1 else 1
            # initialize dask.bag
            bag = db.from_sequence(splits, npartitions=n_chunks)
            # generate enclosed tessellation using dask
            new = bag.map(
                self._tess,
                enclosures,
                buildings,
                inp,
                res,
                threshold,
                unique_id,
            ).compute()

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
        clean_blocks.loc[single, unique_id] = clean_blocks.loc[
            single, "position"
        ].apply(lambda ix: buildings.iloc[res[inp == ix][0]][unique_id])
        return pd.concat(new + [clean_blocks.drop(columns="position")]).reset_index(
            drop=True
        )

    def _tess(
        self,
        ix,
        enclosure,
        buildings,
        query_inp,
        query_res,
        threshold,
        unique_id,
    ):
        poly = enclosure.geometry.array[ix]
        blg = buildings.iloc[query_res[query_inp == ix]]
        within = blg[
            shapely.area(shapely.intersection(blg.geometry.array, poly))
            > (shapely.area(blg.geometry.array) * threshold)
        ].copy()
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
    Generate blocks based on buildings, tessellation, and street network.
    Dissolves tessellation cells based on street-network based polygons.
    Links resulting ID to ``buildings`` and ``tessellation`` as attributes.

    Parameters
    ----------
    tessellation : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    buildings : GeoDataFrame
        A GeoDataFrame containing buildings.
    id_name : str
        The name of the unique blocks ID column to be generated.
    unique_id : str
        The name of the column with the unique ID. If there is none, it can be
        generated with :func:`momepy.unique_id`. This should be the same for
        cells and buildings; ID's should match.

    Attributes
    ----------
    blocks : GeoDataFrame
        A GeoDataFrame containing generated blocks.
    buildings_id : Series
        A Series derived from buildings with block ID.
    tessellation_id : Series
        A Series derived from morphological tessellation with block ID.
    tessellation : GeoDataFrame
        A GeoDataFrame containing original tessellation.
    edges : GeoDataFrame
        A GeoDataFrame containing original edges.
    buildings : GeoDataFrame
        A GeoDataFrame containing original buildings.
    id_name : str
        The name of the unique blocks ID column.
    unique_id : str
        The name of the column with unique ID.

    Examples
    --------
    >>> blocks = mm.Blocks(tessellation_df, streets_df, buildings_df, 'bID', 'uID')
    >>> blocks.blocks.head()
        bID	geometry
    0	1.0	POLYGON ((1603560.078648818 6464202.366899694,...
    1	2.0	POLYGON ((1603457.225976106 6464299.454696888,...
    2	3.0	POLYGON ((1603056.595487018 6464093.903488506,...
    3	4.0	POLYGON ((1603260.943782872 6464141.327631323,...
    4	5.0	POLYGON ((1603183.399594798 6463966.109982309,...
    """

    def __init__(self, tessellation, edges, buildings, id_name, unique_id):
        self.tessellation = tessellation
        self.edges = edges
        self.buildings = buildings
        self.id_name = id_name
        self.unique_id = unique_id

        if id_name in buildings.columns:
            raise ValueError(
                f"'{id_name}' column cannot be in the buildings GeoDataFrame."
            )

        cut = gpd.overlay(
            tessellation,
            gpd.GeoDataFrame(geometry=edges.buffer(0.001)),
            how="difference",
        )
        cut = cut.explode(ignore_index=True) if GPD_10 else cut.explode()

        weights = libpysal.weights.Queen.from_dataframe(cut, silence_warnings=True)
        cut["component"] = weights.component_labels
        buildings_c = buildings.copy()
        buildings_c.geometry = buildings_c.representative_point()  # make points
        if GPD_10:
            centroids_temp_id = gpd.sjoin(
                buildings_c,
                cut[[cut.geometry.name, "component"]],
                how="left",
                predicate="within",
            )
        else:
            centroids_temp_id = gpd.sjoin(
                buildings_c,
                cut[[cut.geometry.name, "component"]],
                how="left",
                op="within",
            )

        cells_copy = tessellation[[unique_id, tessellation.geometry.name]].merge(
            centroids_temp_id[[unique_id, "component"]], on=unique_id, how="left"
        )
        if GPD_10:
            blocks = cells_copy.dissolve(by="component").explode(ignore_index=True)
        else:
            blocks = (
                cells_copy.dissolve(by="component").explode().reset_index(drop=True)
            )
        blocks[id_name] = range(len(blocks))
        blocks = blocks[[id_name, blocks.geometry.name]]

        if GPD_10:
            centroids_w_bl_id2 = gpd.sjoin(
                buildings_c, blocks, how="left", predicate="within"
            )
        else:
            centroids_w_bl_id2 = gpd.sjoin(buildings_c, blocks, how="left", op="within")

        self.buildings_id = centroids_w_bl_id2[id_name]

        cells_m = tessellation[[unique_id]].merge(
            centroids_w_bl_id2[[unique_id, id_name]], on=unique_id, how="left"
        )

        self.tessellation_id = cells_m[id_name]
        self.tessellation_id.index = self.tessellation.index

        self.blocks = blocks


def get_network_id(left, right, network_id, min_size=100, verbose=True):
    """
    Snap each element (preferably building) to the closest
    street network segment and save its ID. Also, adds network ID to elements.

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing objects to snap.
    right : GeoDataFrame
        A GeoDataFrame containing a street network with unique network IDs.
        If there is none, it can be generated with :func:`momepy.unique_id`.
    network_id : str, list, np.array, pd.Series (default None)
        The name of the streets dataframe column, ``np.array``, or ``pd.Series``
        with network unique IDs.
    min_size : int (default 100)
        A minimum size should be a valuee such that if you build a box centered in each
        building centroid with edges of size ``2*min_size``, you know a priori that at
        least one segment is intersected with the box.
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Returns
    -------
    elements_nID : Series
        A Series containing network ID for elements.

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
    infty = 1000000000000
    left = left.copy()
    right = right.copy()

    if not isinstance(network_id, str):
        right["mm_nid"] = network_id
        network_id = "mm_nid"

    buildings_c = left.copy()

    buildings_c[buildings_c.geometry.name] = buildings_c.centroid  # make centroids

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
        d = infty
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

    series = pd.Series(result, index=left.index)

    if series.isnull().any():
        warnings.warn(
            message=(
                "Some objects were not attached to the network. Set larger "
                f"`min_size``. {sum(series.isnull())} affected elements."
            ),
            category=UserWarning,
            stacklevel=2,
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
    Snap each building to the closest street network node on the closest network edge.
    Adds node ID to objects (preferably buildings). Gets ID of edge
    (:func:`momepy.get_network_id` or :func:`get_network_ratio`), and determines
    which of its end points is closer to the building centroid. Pass either ``edge_id``
    with a single value or ``edge_keys`` and ``edge_values`` with ratios.

    Parameters
    ----------
    objects : GeoDataFrame
        A GeoDataFrame containing objects to snap.
    nodes : GeoDataFrame
        A GeoDataFrame containing street nodes with unique node IDs.
        If there is none, it can be generated by :func:`momepy.unique_id`.
    edges : GeoDataFrame
        A GeoDataFrame containing street edges with unique edge IDs and IDs
        of start and end points of each segment. Start and endpoints are default
        outcome of :func:`momepy.nx_to_gdf`.
    node_id : str, list, np.array, pd.Series
        The name of the ``nodes`` dataframe column, ``np.array``,
        or ``pd.Series`` with a unique ID.
    edge_id : str (default None)
        The name of the objects dataframe column with unique edge IDs
        (like an outcome of :func:`momepy.get_network_id`).
    edge_keys : str (default None)
        The name of the objects dataframe column with ``edgeID_keys``
        (like an outcome of :func:`momepy.get_network_ratio`).
    edge_values : str (default None)
        The name of the objects dataframe column with ``edgeID_values``
        (like an outcome of :func:`momepy.get_network_ratio`).
    verbose : bool (default True)
        If ``True``, shows progress bars in loops and indication of steps.

    Returns
    -------
    node_ids : Series
        A Series containing node the ID for objects.
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
                start_id = edge.node_start
                start = nodes.loc[start_id].geometry
                sd = centroid.distance(start)
                end_id = edge.node_end
                end = nodes.loc[end_id].geometry
                ed = centroid.distance(end)
                if sd > ed:
                    results_list.append(end_id)
                else:
                    results_list.append(start_id)

    elif edge_keys is not None and edge_values is not None:
        for edge_i, edge_r, geom in tqdm(
            zip(objects[edge_keys], objects[edge_values], objects.geometry),
            total=objects.shape[0],
            disable=not verbose,
        ):
            edge = edges.iloc[edge_i[edge_r.index(max(edge_r))]]
            start_id = edge.node_start
            start = nodes.loc[start_id].geometry
            sd = geom.distance(start)
            end_id = edge.node_end
            end = nodes.loc[end_id].geometry
            ed = geom.distance(end)
            if sd > ed:
                results_list.append(end_id)
            else:
                results_list.append(start_id)

    series = pd.Series(results_list, index=objects.index)
    return series


def get_network_ratio(df, edges, initial_buffer=500):
    """
    Link polygons to network edges based on the proportion of overlap (if a cell
    intersects more than one edge). Useful if you need to link enclosed tessellation to
    street network. Ratios can be used as weights when linking network-based values
    to cells. For a purely distance-based link use :func:`momepy.get_network_id`.
    Links are based on the integer position of edge (``iloc``).

    Parameters
    ----------
    df : GeoDataFrame
        A GeoDataFrame containing objects to snap (typically enclosed tessellation).
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    initial_buffer : float
        The initial buffer used to link non-intersecting cells.

    Returns
    -------
    result : DataFrame
        The resultant DataFrame.

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

    if not GPD_10:
        raise ImportError("`get_network_ratio` requires geopandas 0.10 or newer.")

    (df_ix, edg_ix), dist = edges.sindex.nearest(
        df.geometry, max_distance=initial_buffer, return_distance=True
    )

    touching = dist < 0.1

    intersections = (
        df.iloc[df_ix[touching]]
        .intersection(edges.buffer(0.0001).iloc[edg_ix[touching]], align=False)
        .reset_index()
    )

    mask = intersections.area > 0.0001

    df_ix_touching = df_ix[touching][mask]
    lengths = intersections[mask].area
    grouped = lengths.groupby(df_ix_touching)
    totals = grouped.sum()
    ints_vect = []
    for name, group in grouped:
        ratios = group / totals.loc[name]
        ints_vect.append(
            {edg_ix[touching][item[0]]: item[1] for item in ratios.items()}
        )

    ratios = pd.Series(ints_vect, index=df.index[list(grouped.groups.keys())])

    near = []
    df_ix_non = df_ix[~touching]
    grouped = pd.Series(dist[~touching]).groupby(df_ix_non)
    for _, group in grouped:
        near.append({edg_ix[~touching][group.idxmin()]: 1.0})

    near = pd.Series(near, index=df.index[list(grouped.groups.keys())])

    ratios = pd.concat([ratios, near])

    nans = df[~df.index.isin(ratios.index)]
    if not nans.empty:
        df_ix, edg_ix = edges.sindex.nearest(
            nans.geometry, return_all=False, max_distance=None
        )
        additional = pd.Series([{i: 1.0} for i in edg_ix], index=nans.index)

        ratios = pd.concat([ratios, additional])

    result = pd.DataFrame()
    result["edgeID_keys"] = ratios.apply(lambda d: list(d.keys()))
    result["edgeID_values"] = ratios.apply(lambda d: list(d.values()))

    return result


def enclosures(
    primary_barriers,
    limit=None,
    additional_barriers=None,
    enclosure_id="eID",
    clip=False,
):
    """
    Generate enclosures based on passed barriers. Enclosures are areas enclosed from
    all sides by at least one type of a barrier. Barriers are typically roads,
    railways, natural features like rivers and other water bodies or coastline.
    Enclosures are a result of polygonization of the  ``primary_barrier`` and ``limit``
    and its subdivision based on additional_barriers.

    Parameters
    ----------
    primary_barriers : GeoDataFrame, GeoSeries
        A GeoDataFrame or GeoSeries containing primary barriers.
        (Multi)LineString geometry is expected.
    limit : GeoDataFrame, GeoSeries, shapely geometry (default None)
        A GeoDataFrame, GeoSeries or shapely geometry containing external limit
        of enclosures, i.e. the area which gets partitioned. If ``None`` is passed,
        the internal area of ``primary_barriers`` will be used.
    additional_barriers : GeoDataFrame
        A GeoDataFrame or GeoSeries containing additional barriers.
        (Multi)LineString geometry is expected.
    enclosure_id : str (default 'eID')
        The name of the ``enclosure_id`` (to be created).
    clip : bool (default False)
        If ``True``, returns enclosures with representative point within the limit
        (if given). Requires ``limit`` composed of Polygon or MultiPolygon geometries.

    Returns
    -------
    enclosures : GeoDataFrame
       A GeoDataFrame containing enclosure geometries and ``enclosure_id``.

    Examples
    --------
    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
    """
    if limit is not None:
        if isinstance(limit, BaseGeometry):
            limit = gpd.GeoSeries([limit])
        if limit.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            limit_b = limit.boundary
        else:
            limit_b = limit
        barriers = pd.concat([primary_barriers.geometry, limit_b.geometry])
    else:
        barriers = primary_barriers
    unioned = barriers.unary_union
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=primary_barriers.crs)

    if additional_barriers is not None:
        if not isinstance(additional_barriers, list):
            raise TypeError(
                "`additional_barriers` expects a list of GeoDataFrames "
                f"or GeoSeries. Got {type(additional_barriers)}."
            )
        additional = pd.concat([gdf.geometry for gdf in additional_barriers])

        inp, res = enclosures.sindex.query_bulk(
            additional.geometry, predicate="intersects"
        )
        unique = np.unique(res)

        new = []

        for i in unique:
            poly = enclosures.array[i]  # get enclosure polygon
            crossing = inp[res == i]  # get relevant additional barriers
            buf = shapely.buffer(poly, 0.01)  # to avoid floating point errors
            crossing_ins = shapely.intersection(
                buf, additional.array[crossing]
            )  # keeping only parts of additional barriers within polygon
            union = shapely.union_all(
                np.append(crossing_ins, shapely.boundary(poly))
            )  # union
            polygons = shapely.get_parts(shapely.polygonize([union]))  # polygonize
            within = shapely.covered_by(
                polygons, buf
            )  # keep only those within original polygon
            new += list(polygons[within])

        final_enclosures = (
            pd.concat(
                [gpd.GeoSeries(enclosures).drop(unique), gpd.GeoSeries(new)]
            ).reset_index(drop=True)
        ).set_crs(primary_barriers.crs)

        final_enclosures = gpd.GeoDataFrame(
            {enclosure_id: range(len(final_enclosures))}, geometry=final_enclosures
        )

    else:
        final_enclosures = gpd.GeoDataFrame(
            {enclosure_id: range(len(enclosures))}, geometry=enclosures
        )

    if clip and limit is not None:
        if not limit.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise TypeError(
                "`limit` requires a GeoDataFrame or GeoSeries with Polygon or "
                "MultiPolygon geometry to be used with `clip=True`."
            )
        _, encl_index = final_enclosures.representative_point().sindex.query_bulk(
            limit.geometry, predicate="contains"
        )
        keep = np.unique(encl_index)
        return final_enclosures.iloc[keep]

    return final_enclosures
