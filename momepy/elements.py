import math
import warnings
from collections import Counter, deque
from enum import Enum

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from joblib import Parallel, delayed
from libpysal.cg import voronoi_frames
from libpysal.graph import Graph
from packaging.version import Version
from pandas import MultiIndex, Series
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

GPD_GE_10 = Version(gpd.__version__) >= Version("1.0dev")
SHPLY_GE_210 = Version(shapely.__version__) >= Version("2.1.0")

__all__ = [
    "morphological_tessellation",
    "enclosed_tessellation",
    "verify_tessellation",
    "get_nearest_street",
    "get_nearest_node",
    "generate_blocks",
    "buffered_limit",
    "enclosures",
    "get_network_ratio",
]


def morphological_tessellation(
    geometry: GeoSeries | GeoDataFrame,
    clip: str | shapely.Geometry | GeoSeries | GeoDataFrame | None = "bounding_box",
    shrink: float = 0.4,
    segment: float = 0.5,
    simplify: bool = True,
    **kwargs,
) -> GeoDataFrame:
    """Generate morphological tessellation.

    Morpohological tessellation is a method to divide space into cells based on
    building footprints and Voronoi tessellation. The function wraps
    :func:`libpysal.cg.voronoi_frames` and provides customized default parameters
    following :cite:`fleischmann2020`.

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

    All three types can be tested using :class:`momepy.CheckTessellationInput`.

    See :cite:`fleischmann2020` for details of implementation.

    Parameters
    ----------
    geometry : GeoSeries | GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    clip : str | shapely.Geometry | GeoSeries | GeoDataFrame | None
        Polygon used to clip the Voronoi polygons, by default "bounding_box". You can
        pass any option accepted by :func:`libpysal.cg.voronoi_frames` or geopandas
        object that will be automatically unioned.
    shrink : float, optional
        The distance for negative buffer to generate space between adjacent polygons).
        By default 0.4
    segment : float, optional
        The maximum distance between points after discretization. By default 0.5
    simplify: bool, optional
        Whether to attempt to simplify the resulting tesselation boundaries with
        ``shapely.coverage_simplify``. By default True.
    **kwargs
        Additional keyword arguments pased to libpysal.cg.voronoi_frames, such as
        ``grid_size``.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an index matching the index of input geometry

    See also
    --------
    momepy.enclosed_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a limit used to clip the extent:

    >>> limit = momepy.buffered_limit(buildings, buffer="adaptive")

    Generate tessellation:

    >>> momepy.morphological_tessellation(buildings).head()
                                                geometry
    0  POLYGON ((1603536.56 6464392.264, 1603541.262 ...
    1  POLYGON ((1603167.679 6464323.194, 1603167.552...
    2  POLYGON ((1603078.787 6464172.1, 1603077.665 6...
    3  POLYGON ((1603070.306 6464154.611, 1603070.081...
    4  POLYGON ((1603083.134 6464103.971, 1603077.387...

    """
    if simplify and not SHPLY_GE_210:
        # TODO: remove the keyword and do simplification by default once it is
        # safe to pin shapely 2.1
        raise ImportError(
            "`simplify=True` requires shapely 2.1 or higher. "
            "Update shapely or set `simplify` to False."
        )

    if isinstance(geometry.index, MultiIndex):
        raise ValueError(
            "MultiIndex is not supported in `momepy.morphological_tessellation`."
        )

    if isinstance(clip, GeoSeries | GeoDataFrame):
        clip = clip.union_all() if GPD_GE_10 else clip.unary_union

    mt = voronoi_frames(
        geometry,
        clip=clip,
        shrink=shrink,
        segment=segment,
        return_input=False,
        as_gdf=True,
        **kwargs,
    )
    if simplify:
        mt.geometry = shapely.coverage_simplify(
            mt.geometry, tolerance=segment / 2, simplify_boundary=False
        )
    return mt


def enclosed_tessellation(
    geometry: GeoSeries | GeoDataFrame,
    enclosures: GeoSeries | GeoDataFrame,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    simplify: bool = True,
    n_jobs: int = -1,
    inner_barriers: GeoSeries | GeoDataFrame = None,
    cell_size: float = 1,
    neighbor_mode: str = "moore",
    **kwargs,
) -> GeoDataFrame:
    """Generate enclosed tessellation

    Enclosed tessellation is an enhanced :func:`morphological_tessellation`, based on
    predefined enclosures and building footprints. We can see enclosed tessellation as
    two-step partitioning of space based on building footprints and boundaries (e.g.
    street network, railway). Original morphological tessellation is used under the hood
    to partition each enclosure.

    Tessellation requires data of relatively high level of precision and there are three
    particular patterns causing issues:

    1. Features will collapse into empty polygon - these
       do not have tessellation cell in the end.
    2. Features will split into MultiPolygons - in some cases,
       features with narrow links between parts split into two during 'shrinking'.
       In most cases that is not an issue and the resulting tessellation is correct
       anyway, but sometimes this results in a cell being a MultiPolygon, which is
       not correct.
    3. Overlapping features - features which overlap even
       after 'shrinking' cause invalid tessellation geometry.

    All three types can be tested using :class:`momepy.CheckTessellationInput`.

    The index of the resulting GeoDataFrame links the input buildings with the output
    geometry. Enclosures with no buildings are also included in the output with negative
    index. Ensure that the input geometry has unique non-negative index for this to work
    correctly.

    Parameters
    ----------
    geometry : GeoSeries | GeoDataFrame
        A GeoDataFrame or GeoSeries containing buildings to tessellate the space around.
    enclosures : GeoSeries | GeoDataFrame
        The enclosures geometry, which can be generated using :func:`momepy.enclosures`.
    shrink : float, optional
        The distance for negative buffer to generate space between adjacent polygons).
        By default 0.4
    segment : float, optional
        The maximum distance between points after discretization. By default 0.5
    threshold : float, optional
        The minimum threshold for a building to be considered within an enclosure.
        Threshold is a ratio of building area which needs to be within an enclosure to
        inlude it in the tessellation of that enclosure. Resolves sliver geometry
        issues. If None, the check is skipped and all intersecting buildings are
        considered. By default 0.05
    simplify: bool, optional
        Whether to attempt to simplify the resulting tesselation boundaries with
        ``shapely.coverage_simplify``. By default True.
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using all available cores.
        By default -1
    inner_barriers: GeoSeries | GeoDataFrame, optional
        Barriers that should be included in the tessellation process. By passing
        inner barriers, tessellation will be derived using less performant
        Cellular Automata implememtation but it will recognise dangling barries as valid
        limits of the cell growth. See the user guide for details. By default None.
    cell_size : float, optional
        Grid cell size when ``inner_barriers`` is not None. Otherwise ignored.
        By default 1.0
    neighbor_mode : str, optional
        Choice of neighbor connectivity ('moore' or 'neumann') when ``inner_barriers``
        is not None. Otherwise ignored.. By default 'moore'.
    **kwargs
        Additional keyword arguments pased to libpysal.cg.voronoi_frames, such as
        ``grid_size``.

    Warnings
    --------
    Due to the floating point precision issues in clipping the tessellation cells to the
    extent of their parental enclosures, the result does not form a precise polygonal
    coverage. To build a contiguity graph, use fuzzy contiguity builder with a small
    buffer, e.g.::

        from libpysal import graph

        graph.Graph.build_fuzzy_contiguity(tessellation, buffer=1e-6)

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with an index matching the index of input geometry and a column
        matching the index of input enclosures.

    See also
    --------
    momepy.enclosures
    momepy.morphological_tessellation
    momepy.CheckTessellationInput
    momepy.verify_tessellation

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Generate enclosures:

    >>> enclosures = momepy.enclosures(streets)

    Generate tessellation:

    >>> momepy.enclosed_tessellation(buildings, enclosures).head()
                                                  geometry  enclosure_index
    0    POLYGON ((1603546.697 6464383.596, 1603585.64 ...                0
    113  POLYGON ((1603517.131 6464349.296, 1603546.697...                0
    114  POLYGON ((1603517.87 6464285.864, 1603515.152 ...                0
    125  POLYGON ((1603586.269 6464256.691, 1603581.813...                0
    126  POLYGON ((1603499.92 6464243.917, 1603493.299 ...                0
    """

    if simplify and not SHPLY_GE_210:
        # TODO: remove the keyword and do simplification by default once it is
        # safe to pin shapely 2.1
        raise ImportError(
            "`simplify=True` requires shapely 2.1 or higher. "
            "Update shapely or set `simplify` to False."
        )

    if isinstance(geometry.index, MultiIndex):
        raise ValueError(
            "MultiIndex is not supported in `momepy.enclosed_tessellation`."
        )

    # convert to GeoDataFrame and add position (we will need it later)
    enclosures = enclosures.geometry.to_frame()
    enclosures["position"] = range(len(enclosures))

    # preserve index name if exists
    index_name = enclosures.index.name
    if not index_name:
        index_name = "enclosure_index"
    enclosures[index_name] = enclosures.index

    # figure out which enlosures contain which buildings
    inp, res = geometry.sindex.query(enclosures.geometry, predicate="intersects")

    # find out which enclosures contain one and multiple buildings
    unique, counts = np.unique(inp, return_counts=True)
    splits = unique[counts > 1]
    single = unique[counts == 1]
    altered = unique[counts > 0]

    # prepare input for parallel processing
    tuples = [
        (
            enclosures.index[i],  # enclosure index
            enclosures.geometry.iloc[i],  # enclosure geometry
            geometry.iloc[res[inp == i]],  # buildings within the enclosure
        )
        for i in splits
    ]

    # generate tessellation in parallel
    new = Parallel(n_jobs=n_jobs)(
        delayed(_tess)(
            *t,
            threshold,
            shrink,
            segment,
            index_name,
            simplify,
            inner_barriers,
            cell_size,
            neighbor_mode,
            kwargs,
        )
        for t in tuples
    )

    new_df = pd.concat(new, axis=0)

    # some enclosures had building intersections that did not meet the threshold
    if -1 in new_df.index:
        unchanged_in_new = new_df.loc[[-1]]
        new_df = new_df.drop(-1)
        clean_blocks = pd.concat(
            [
                enclosures.drop(enclosures.index[altered]).drop(columns="position"),
                unchanged_in_new,
            ]
        )
    else:
        clean_blocks = enclosures.drop(enclosures.index[altered]).drop(
            columns="position"
        )

    # assign negative index to enclosures with no buildings
    clean_blocks.index = range(-len(clean_blocks), 0, 1)

    # get building index for enclosures with single building
    singles = enclosures.iloc[single]
    singles.index = singles.position.loc[singles.index].apply(
        lambda ix: geometry.iloc[res[inp == ix]].index[0]
    )
    # combine results
    return pd.concat([new_df, singles.drop(columns="position"), clean_blocks])


def _tess(
    ix,
    poly,
    blg,
    threshold,
    shrink,
    segment,
    enclosure_id,
    to_simplify,
    inner_barriers,
    cell_size,
    neighbor_mode,
    kwargs,
):
    """Generate tessellation for a single enclosure. Helper for enclosed_tessellation.

    Parameters
    ----------
    ix : int
        Enclosure index.
    poly : shapely.Geometry
        Enclosure geometry.
    blg : GeoSeries | GeoDataFrame
        Buildings within the enclosure.
    threshold : float
        Threshold for building inclusion.
    shrink : float
        Shrink distance for tessellation.
    segment : float
        Segmentation distance.
    enclosure_id : str
        Column name for enclosure ID.
    inner_barriers : GeoSeries | GeoDataFrame
        Inner barriers for tessellation.
    cell_size : float
        Grid cell size when inner_barriers is not None.
    neighbor_mode : str
        Choice of neighbor connectivity ('moore' or 'neumann')
        when inner_barriers is not None.

    Returns
    -------
    GeoDataFrame
        Tessellation for the enclosure.
    """
    # check if threshold is set and filter buildings based on the threshold
    if threshold:
        blg = blg[
            shapely.area(shapely.intersection(blg.geometry.array, poly))
            > (shapely.area(blg.geometry.array) * threshold)
        ]

    if len(blg) > 1:
        if inner_barriers is None or inner_barriers.empty:
            tess = voronoi_frames(
                blg,
                clip=poly,
                shrink=shrink,
                segment=segment,
                return_input=False,
                as_gdf=True,
                **kwargs,
            )
            tolerance = segment / 2

        else:
            tess = _voronoi_by_ca(
                seed_geoms=blg,
                barrier_geoms=poly,
                cell_size=cell_size,
                neighbor_mode=neighbor_mode,
                barriers_for_inner=inner_barriers,
            )
            # torelance set as the square root of the isosceles right triangle with 2
            # cells_size edges
            tolerance = ((2 * cell_size) ** 2 / 2) ** 0.5

        if to_simplify:
            tess.geometry = shapely.coverage_simplify(
                tess.geometry, tolerance=tolerance, simplify_boundary=False
            )

        tess[enclosure_id] = ix
        return tess

    ## in case a single building is left in blg
    if len(blg) == 1:
        assigned_ix = blg.index[0]
    else:
        assigned_ix = -1

    return GeoDataFrame(
        {enclosure_id: ix},
        geometry=[poly],
        index=[assigned_ix],
        crs=blg.crs,
    )


def _voronoi_by_ca(
    seed_geoms: GeoSeries | GeoDataFrame,
    barrier_geoms: GeoSeries | GeoDataFrame,
    cell_size: float = 1.0,
    neighbor_mode: str = "moore",
    barriers_for_inner: GeoSeries | GeoDataFrame = None,
) -> GeoDataFrame:
    """
    Generate an aggregated Voronoi tessellation as a GeoDataFrame via cellular automata.

    This unified function performs the following:

    - Ensures that the CRS of seed and barrier geometries are aligned.
    - Combines inner barriers with the enclosure.
    - Computes grid bounds covering both seed and barrier geometries.
    - Marks barrier cells using a prepared geometry for fast intersection.
    - Seeds the grid with seed geometries.
    - Propagates seed values via a BFS expansion that respects barriers.
    - Uses a voting mechanism to finalize boundary cell assignments.
    - Converts grid cells into a GeoDataFrame and dissolves adjacent cells
        with the same seed id.
    - Clips the output cells by the barrier boundaries.

    Parameters
    ----------
    seed_geoms : GeoSeries | GeoDataFrame
        GeoDataFrame containing seed features.
    barrier_geoms : GeoSeries | GeoDataFrame
        GeoDataFrame containing barrier features or a shapely Polygon.
    cell_size : float, optional
        Grid cell size. By default 1.0
    neighbor_mode : str, optional
        Choice of neighbor connectivity ('moore' or 'neumann'). By default 'moore'.
    barriers_for_inner : GeoSeries | GeoDataFrame, optional
        GeoDataFrame containing inner barriers to be included. By default None


    Returns
    -------
    GeoDataFrame
        A GeoDataFrame representing the aggregated Voronoi tessellation, clipped by
        barriers.
    """
    # Get inner barriers as intersection or containment of the barrier_geoms
    inner_barriers = _get_inner_barriers(barrier_geoms, barriers_for_inner)

    # Handle barrier_geoms if it is a Polygon or MultiPolygon
    if barrier_geoms.geom_type == "Polygon":
        # Take buffer of polygon and extract its exterior boundary (10 cells)
        barrier_geoms_buffered = GeoSeries(
            [barrier_geoms.buffer(10 * cell_size).exterior], crs=seed_geoms.crs
        )
        barrier_geoms = GeoSeries([barrier_geoms], crs=seed_geoms.crs)

    elif barrier_geoms.geom_type == "MultiPolygon":
        # Process each polygon: take buffer then exterior boundary
        # (to ensure there's no gap between enclosures)
        barrier_geoms_buffered = GeoSeries(
            shapely.buffer(
                shapely.get_exterior(shapely.get_parts(barrier_geoms)), 10 * cell_size
            ),
            crs=seed_geoms.crs,
        )
        barrier_geoms = GeoSeries(barrier_geoms, crs=seed_geoms.crs)

    else:
        raise ValueError("Enclosure must be a Polygon or MultiPolygon")

    outer_union = barrier_geoms_buffered.union_all()

    # Compute inner barriers union if available
    if inner_barriers is not None and not inner_barriers.empty:
        inner_union = inner_barriers.union_all()
    else:
        inner_union = None

    # Combine outer barrier with inner barriers
    if outer_union and inner_union:
        prep_barrier = shapely.union(outer_union, inner_union)
    elif outer_union:
        prep_barrier = outer_union
    elif inner_union:
        prep_barrier = inner_union
    else:
        prep_barrier = None

    # Compute grid bounds
    origin, grid_width, grid_height = _get_grid_bounds(
        seed_geoms, barrier_geoms_buffered, cell_size
    )

    # Initialize grid states with UNKNOWN values.
    states = np.full((grid_height, grid_width), CellState.UNKNOWN.value, dtype=int)

    xs, ys = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    cell_polys = GeoSeries(
        [
            _get_cell_polygon(x, y, cell_size, origin)
            for x, y in zip(xs.flatten(), ys.flatten(), strict=True)
        ]
    )

    # Identify barrier cells in the grid
    if prep_barrier is not None:
        barrier_mask = cell_polys.intersects(prep_barrier).values.reshape(
            grid_height, grid_width
        )
    else:
        barrier_mask = np.zeros((grid_height, grid_width), dtype=bool)
    states[barrier_mask] = CellState.BARRIER.value

    # Seed the grid with seed geometries.
    for site_id, geom in enumerate(seed_geoms.geometry):
        if not geom.is_empty:
            cells = _geom_to_cells(geom, origin, cell_size, grid_width, grid_height)
            valid_cells = [
                (x, y) for x, y in cells if states[y, x] == CellState.UNKNOWN.value
            ]
            if valid_cells:
                indices = np.array(valid_cells)
                states[indices[:, 1], indices[:, 0]] = site_id

    # Initialize the BFS queue with all seeded cells’ neighbors.
    queue = deque()
    seed_indices = np.argwhere(states >= 0)
    for y, x in seed_indices:
        _enqueue_neighbors(x, y, states, grid_width, grid_height, neighbor_mode, queue)

    # Process BFS to propagate seed values.
    while queue:
        # Dequeue the current cell and skip if it is not a frontier cell.
        x_current, y_current = queue.popleft()
        if states[y_current, x_current] != CellState.FRONTIER.value:
            continue

        # Get neighbor cells that were already assigned a seed id or still unknown
        # (state >= 0).
        # Note that boundary or barrier cells are skipped (state < 0).
        neighbor_seeds = [
            states[ny, nx]
            for nx, ny in _get_neighbors(
                x_current, y_current, grid_width, grid_height, mode=neighbor_mode
            )
            if states[ny, nx] >= 0
        ]
        if not neighbor_seeds:
            continue

        # Assign as a boundary if multiple seed ids are found.
        if len(set(neighbor_seeds)) > 1:
            states[y_current, x_current] = CellState.BOUNDARY.value
        # If not, equeue neighbor cells for further propagation.
        else:
            assigned_seed = set(neighbor_seeds).pop()
            states[y_current, x_current] = assigned_seed
            _enqueue_neighbors(
                x_current,
                y_current,
                states,
                grid_width,
                grid_height,
                neighbor_mode,
                queue,
            )

    # Post-process barrier and boundary cells using a voting mechanism.
    states = _assign_adjacent_seed_cells(states, neighbor_mode)

    # Create grid cell polygons and build a GeoDataFrame.
    xs, ys = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    grid_polys = [
        _get_cell_polygon(x, y, cell_size, origin)
        for x, y in zip(xs.flatten(), ys.flatten(), strict=True)
    ]
    grid_gdf = GeoDataFrame(
        {"site_id": states.flatten()}, geometry=grid_polys, crs=seed_geoms.crs
    )

    # Include only cells with valid seed assignments and dissolve contiguous regions.
    grid_gdf = (
        grid_gdf[grid_gdf["site_id"] >= 0]
        .dissolve(by="site_id")
        .reset_index()
        .drop("site_id", axis=1)
    )

    # Clip by barriers
    if barrier_geoms is not None and (not barrier_geoms.empty):
        # Create a union of the barrier geometries.
        barrier_union = barrier_geoms.union_all()
        # If the barrier union is not a polygon (e.g., it's a MultiLineString),
        # polygonize it.
        if not isinstance(
            barrier_union, shapely.geometry.Polygon | shapely.geometry.MultiPolygon
        ):
            barrier_polys = shapely.get_parts(shapely.polygonize(barrier_union))
            if not barrier_polys.empty:
                barrier_union = shapely.union_all(barrier_polys)
        # Clip each polygon in the grid using the barrier boundary.
        grid_gdf["geometry"] = grid_gdf["geometry"].intersection(barrier_union)

    # cleanup possible collections
    def sanitize_geometry(geom):
        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            return geom
        elif geom.geom_type == "GeometryCollection":
            parts = shapely.get_parts(geom)
            valid_parts = [
                p for p in parts if p.geom_type in ["Polygon", "MultiPolygon"]
            ]
            if len(valid_parts) == 0:
                return None
            elif len(valid_parts) == 1:
                return valid_parts[0]
            else:
                return shapely.MultiPolygon(valid_parts)
        else:
            # Drop points, lines, etc.
            return None

    grid_gdf["geometry"] = grid_gdf["geometry"].apply(sanitize_geometry)
    grid_gdf = grid_gdf.dropna(subset=["geometry"])
    return grid_gdf


def _get_inner_barriers(enclosure, barriers):
    """Get inner barriers that intersect or are contained within an enclosure.

    Parameters
    ----------
    enclosure : GeoSeries | GeoDataFrame
        The enclosure geometry.
    barriers : GeoDataFrame
        The barriers GeoDataFrame.

    Returns
    -------
    shapely.Polygon
        A single Polygon combining the enclosure and any intersecting barriers.
    """
    # Clip those segments to stay within the enclosure
    inner_barriers = gpd.clip(barriers, enclosure)

    # Only keep the geometry which is within the enclosure
    inner_barriers = inner_barriers[inner_barriers.within(enclosure)]

    return inner_barriers


class CellState(Enum):
    """
    Enumeration of cell states for grid processing for improving the readability,
    instead of integers.

    Attributes
    ----------
    UNKNOWN : int
        Cell has not been processed.
    BOUNDARY : int
        Cell is at a junction between different seed regions.
    BARRIER : int
        Cell originally designated as a barrier.
    FRONTIER : int
        Cell queued for BFS expansion.
    """

    UNKNOWN = -1
    BOUNDARY = -2
    BARRIER = -3
    FRONTIER = -4


def _get_cell_polygon(
    x_idx: int, y_idx: int, cell_size: float = 1.0, origin: tuple[float, float] = (0, 0)
) -> shapely.geometry.Polygon:
    """
    Generate a grid cell polygon based on the given indices, cell size, and origin.
    """
    ox, oy = origin
    return shapely.Polygon(
        [
            (ox + x_idx * cell_size, oy + y_idx * cell_size),
            (ox + (x_idx + 1) * cell_size, oy + y_idx * cell_size),
            (ox + (x_idx + 1) * cell_size, oy + (y_idx + 1) * cell_size),
            (ox + x_idx * cell_size, oy + (y_idx + 1) * cell_size),
        ]
    )


def _get_neighbors(
    x: int, y: int, max_x: int, max_y: int, mode: str = "moore"
) -> list[tuple[int, int]]:
    """
    Retrieve valid neighboring cell indices based on connectivity.

    Parameters
    ----------
    x : int
        The current cell x index.
    y : int
        The current cell y index.
    max_x : int
        The maximum x dimension of the grid.
    max_y : int
        The maximum y dimension of the grid.
    mode : str, optional
        The connectivity mode, "moore" for 8-connected or "neumann" for 4-connected
        neighbors. By default "moore".

    Returns
    -------
    list[tuple[int, int]]
        A list of (x, y) tuples for valid neighbor indices.
    """
    neighbor_dirs = {
        "moore": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        "neumann": [(0, -1), (-1, 0), (1, 0), (0, 1)],
    }
    directions = neighbor_dirs.get(mode)
    if directions is None:
        raise ValueError("Invalid neighbor_mode: choose 'moore' or 'neumann'")
    return [
        (x + dx, y + dy)
        for dx, dy in directions
        if 0 <= x + dx < max_x and 0 <= y + dy < max_y
    ]


def _get_grid_bounds(
    seed_geoms: GeoSeries | GeoDataFrame,
    barrier_geoms: GeoSeries | GeoDataFrame,
    cell_size: float,
) -> tuple[tuple[float, float], int, int]:
    seed_bounds = seed_geoms.total_bounds  # [xmin, ymin, xmax, ymax]
    barrier_bounds = barrier_geoms.total_bounds

    xmin = min(seed_bounds[0], barrier_bounds[0])
    ymin = min(seed_bounds[1], barrier_bounds[1])
    xmax = max(seed_bounds[2], barrier_bounds[2])
    ymax = max(seed_bounds[3], barrier_bounds[3])
    # expand bounds by 1 cell in each direction
    new_xmin = xmin - cell_size
    new_ymin = ymin - cell_size
    new_xmax = xmax + cell_size
    new_ymax = ymax + cell_size
    grid_width = math.ceil((new_xmax - new_xmin) / cell_size)
    grid_height = math.ceil((new_ymax - new_ymin) / cell_size)
    return (new_xmin, new_ymin), grid_width, grid_height


def _geom_to_cells(
    geom: shapely.geometry,
    origin: tuple[float, float],
    cell_size: float,
    grid_width: int,
    grid_height: int,
) -> list[tuple[int, int]]:
    """
    Determine grid cell indices that intersect the given geometry.
    """
    if isinstance(geom, shapely.geometry.Point):
        sx = int((geom.x - origin[0]) // cell_size)
        sy = int((geom.y - origin[1]) // cell_size)
        return [(sx, sy)] if 0 <= sx < grid_width and 0 <= sy < grid_height else []

    else:
        minx, miny, maxx, maxy = geom.bounds
        start_x = max(0, int((minx - origin[0]) // cell_size))
        start_y = max(0, int((miny - origin[1]) // cell_size))
        end_x = min(grid_width, int(math.ceil((maxx - origin[0]) / cell_size)))
        end_y = min(grid_height, int(math.ceil((maxy - origin[1]) / cell_size)))

        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)
        xx, yy = np.meshgrid(x_range, y_range)
        candidate_polys = GeoSeries(
            [
                _get_cell_polygon(x, y, cell_size, origin)
                for x, y in zip(xx.flatten(), yy.flatten(), strict=True)
            ]
        )
        mask = candidate_polys.intersects(geom)
        return list(zip(xx.flatten()[mask], yy.flatten()[mask], strict=True))


def _enqueue_neighbors(
    x: int,
    y: int,
    states: np.ndarray,
    grid_width: int,
    grid_height: int,
    neighbor_mode: str,
    queue: deque,
) -> None:
    """
    Enqueue valid neighboring cells for BFS expansion.
    """
    for nx, ny in _get_neighbors(x, y, grid_width, grid_height, mode=neighbor_mode):
        if states[ny, nx] == CellState.UNKNOWN.value:
            states[ny, nx] = CellState.FRONTIER.value
            queue.append((nx, ny))


def _assign_adjacent_seed_cells(
    states: np.ndarray, neighbor_mode: str = "moore"
) -> np.ndarray:
    """
    Reassign border and barrier cells to the proximate seed areas using a voting
    mechanism.
    """
    new_states = states.copy()
    indices = np.argwhere(
        np.isin(states, [CellState.BARRIER.value, CellState.BOUNDARY.value])
    )
    grid_height, grid_width = states.shape

    for y, x in indices:
        neighbor_seeds = [
            states[ny, nx]
            for nx, ny in _get_neighbors(
                x, y, grid_width, grid_height, mode=neighbor_mode
            )
            if states[ny, nx] >= 0
        ]
        if neighbor_seeds:
            cnt = Counter(neighbor_seeds)
            # In case of ties, choose the smaller seed id.
            chosen_seed = min(cnt.items(), key=lambda item: (-item[1], item[0]))[0]
            new_states[y, x] = chosen_seed
    return new_states


def verify_tessellation(tessellation, geometry):
    """Check whether result matches buildings and contains only Polygons.

    Checks if the generated tessellation fully matches the input buildings, i.e. if
    there are all building indices present in the tessellation. Also checks if there are
    any MultiPolygons present in the tessellation. The former is often caused by
    buildings collapsing during the generation process, the latter is usually caused by
    errors in the input geometry, overlapping buildings, or narrow links between parts
    of the building.

    Parameters
    ----------
    tessellation : GeoSeries | GeoDataFrame
        tessellation geometry
    geometry : GeoSeries | GeoDataFrame
        building geometry used to generate tessellation

    Returns
    -------
    tuple(excluded, multipolygons)
        Tuple of indices of building IDs not present in tessellations and MultiPolygons.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")

    Define a limit used to clip the extent:

    >>> limit = momepy.buffered_limit(buildings, buffer="adaptive")

    Generate tessellation:

    >>> tessellation = momepy.morphological_tessellation(buildings)

    Verify the result.

    >>> excluded, multipolygons = momepy.verify_tessellation(tessellation, buildings)
    """

    if isinstance(geometry.index, MultiIndex) or isinstance(
        tessellation.index, MultiIndex
    ):
        raise ValueError("MultiIndex is not supported in `momepy.verify_tessellation`.")

    # check against input layer
    ids_original = geometry.index
    ids_generated = tessellation.index
    collapsed = pd.Index([])
    if len(ids_original) != len(ids_generated):
        collapsed = ids_original.difference(ids_generated)
        warnings.warn(
            message=(
                "Tessellation does not fully match buildings. "
                f"{len(collapsed)} element(s) disappeared "
                f"during generation. Index of the affected elements: {collapsed}."
            ),
            category=UserWarning,
            stacklevel=2,
        )

    # check MultiPolygons - usually caused by error in input geometry
    multipolygons = tessellation[
        tessellation.geometry.geom_type == "MultiPolygon"
    ].index
    if len(multipolygons) > 0:
        warnings.warn(
            message=(
                "Tessellation contains MultiPolygon elements. Initial "
                "objects should  be edited. Index of affected "
                f"elements: {list(multipolygons)}."
            ),
            category=UserWarning,
            stacklevel=2,
        )
    return collapsed, multipolygons


def get_nearest_street(
    buildings: GeoSeries | GeoDataFrame,
    streets: GeoSeries | GeoDataFrame,
    max_distance: float | None = None,
) -> Series:
    """Identify the nearest street for each building.

    Parameters
    ----------
    buildings : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of buildings
    streets : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of streets
    max_distance : float | None, optional
        Maximum distance within which to query for nearest street. Must be
        greater than 0. By default None, indicating no distance limit. Note that it is
        advised to set a limit to avoid long processing times.

    Notes
    -----
    In case of multiple streets within the same distance, only one is returned.

    Returns
    -------
    np.ndarray
        array containing the index of the nearest street for each building

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Get street index.

    >>> momepy.get_nearest_street(buildings, streets)
    0       0.0
    1      33.0
    2      10.0
    3       8.0
    4       8.0
        ...
    139    34.0
    140    32.0
    141    21.0
    142    16.0
    143    19.0
    Length: 144, dtype: float64
    """
    blg_idx, str_idx = streets.sindex.nearest(
        buildings.geometry, return_all=False, max_distance=max_distance
    )

    ids = pd.Series(None, index=buildings.index, dtype=streets.index.dtype)

    ids.iloc[blg_idx] = streets.index[str_idx]
    return ids


def get_nearest_node(
    buildings: GeoSeries | GeoDataFrame,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    nearest_edge: Series,
) -> Series:
    """Identify the nearest node for each building.

    Snap each building to the closest street network node on the closest network edge.
    This assumes that the nearest street network edge has already been identified using
    :func:`get_nearest_street`.

    The ``edges`` and ``nodes`` GeoDataFrames are expected to be an outcome of
    :func:`momepy.nx_to_gdf` or match its structure with ``["node_start", "node_end"]``
    columns and their meaning.


    Parameters
    ----------
    buildings : GeoSeries | GeoDataFrame
        GeoSeries or GeoDataFrame of buildings.
    nodes : GeoDataFrame
        A GeoDataFrame containing street nodes.
    edges : GeoDataFrame
        A GeoDataFrame containing street edges with ``["node_start", "node_end"]``
        columns marking start and end nodes of each edge. These are the default
        outcome of :func:`momepy.nx_to_gdf`.
    nearest_edge : Series
        A Series aligned with ``buildings`` containing the information on the nearest
        street edge. Matches the outcome of :func:`get_nearest_street`.

    Returns
    -------
    Series

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Pass an object via ``networkx`` to get the nodes and necessary information.

    >>> G = momepy.gdf_to_nx(streets)
    >>> nodes, edges = momepy.nx_to_gdf(G)

    Get nearest edge:

    >>> buildings["edge_index"] = momepy.get_nearest_street(buildings, edges)

    Get nearest node:

    >>> momepy.get_nearest_node(buildings, nodes, edges, buildings["edge_index"])
    0       0.0
    1       9.0
    2      11.0
    3      11.0
    4      11.0
        ...
    139     1.0
    140    20.0
    141    15.0
    142     2.0
    143    22.0
    Length: 144, dtype: float64
    """

    if (
        isinstance(buildings.index, MultiIndex)
        or isinstance(nearest_edge.index, MultiIndex)
        or isinstance(nodes.index, MultiIndex)
        or isinstance(edges.index, MultiIndex)
    ):
        raise ValueError("MultiIndex is not supported in `momepy.get_nearest_node`.")

    # treat possibly missing edge index
    a = np.empty(len(buildings))
    na_mask = np.isnan(nearest_edge)
    a[na_mask] = np.nan

    streets = edges.loc[nearest_edge[~na_mask]]
    starts = nodes.loc[streets["node_start"]].distance(buildings[~na_mask], align=False)
    ends = nodes.loc[streets["node_end"]].distance(buildings[~na_mask], align=False)
    mask = starts.values > ends.values
    r = starts.index.to_numpy(copy=True)
    r[mask] = ends.index[mask]

    a[~na_mask] = r
    return pd.Series(a, index=buildings.index)


def generate_blocks(
    tessellation: GeoDataFrame, edges: GeoDataFrame, buildings: GeoDataFrame
) -> tuple[GeoDataFrame, Series]:
    """
    Generate blocks based on buildings, tessellation, and street network.
    Dissolves tessellation cells based on street-network based polygons.
    Links resulting ID to ``tessellation`` and returns
    ``blocks`` and ``tessellation`` ids.

    Parameters
    ----------
    tessellation : GeoDataFrame
        A GeoDataFrame containing morphological tessellation.
    edges : GeoDataFrame
        A GeoDataFrame containing a street network.
    buildings : GeoDataFrame
        A GeoDataFrame containing buildings.

    Notes
    -----
    This function assumes morphological tessellation and 1:1 relationship
    between buildings and cells. Tesselation cells that do not have buildings
    can break the functionality.

    Returns
    -------
    blocks : GeoDataFrame
        A GeoDataFrame containing generated blocks.
    tessellation_ids : Series
        A Series derived from morphological tessellation with block ID.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> streets = geopandas.read_file(path, layer="streets")

    Generate tessellation:

    >>> tessellation = momepy.morphological_tessellation(buildings)
    >>> tessellation.head()
                                                geometry
    0  POLYGON ((1603536.56 6464392.264, 1603541.262 ...
    1  POLYGON ((1603167.679 6464323.194, 1603167.552...
    2  POLYGON ((1603078.787 6464172.1, 1603077.665 6...
    3  POLYGON ((1603070.306 6464154.611, 1603070.081...
    4  POLYGON ((1603083.134 6464103.971, 1603077.387...

    >>> blocks, tessellation_id = momepy.generate_blocks(
    ...     tessellation, streets, buildings
    ... )
    >>> blocks.head()
                                                geometry
    0  POLYGON ((1603421.741 6464282.377, 1603415.23 ...
    1  POLYGON ((1603485.548 6464217.177, 1603483.228...
    2  POLYGON ((1603314.034 6464117.593, 1603295.424...
    3  POLYGON ((1602992.334 6464131.13, 1602992.334 ...
    4  POLYGON ((1602992.334 6463992.499, 1602992.334...

    ``tessellation_id`` can be directly assigned to its
    respective parental DataFrame directly.

    >>> tessellation["block_id"] = tessellation_id
    """

    if (
        isinstance(buildings.index, MultiIndex)
        or isinstance(tessellation.index, MultiIndex)
        or isinstance(edges.index, MultiIndex)
    ):
        raise ValueError("MultiIndex is not supported in `momepy.generate_blocks`.")
    id_name: str = "bID"

    # slice the tessellations by the street network
    cut = gpd.overlay(
        tessellation,
        gpd.GeoDataFrame(geometry=edges.buffer(0.001)),
        how="difference",
    )
    cut = cut.explode(ignore_index=True)
    # touching tessellations form a block
    weights = Graph.build_contiguity(cut, rook=False)
    cut["component"] = weights.component_labels

    # generate block geometries
    buildings_c = buildings.copy()
    buildings_c.geometry = buildings_c.representative_point()  # make points
    centroids_temp_id = gpd.sjoin(
        buildings_c,
        cut[[cut.geometry.name, "component"]],
        how="left",
        predicate="within",
    )
    cells_copy = tessellation[[tessellation.geometry.name]].merge(
        centroids_temp_id[["component"]], right_index=True, left_index=True, how="left"
    )
    blocks = cells_copy.dissolve(by="component").explode(ignore_index=True)

    # assign block ids to buildings and tessellations
    centroids_w_bl_id2 = gpd.sjoin(buildings_c, blocks, how="left", predicate="within")
    buildings_id = centroids_w_bl_id2["index_right"]
    buildings_id.name = id_name
    cells_m = tessellation.merge(
        buildings_id, left_index=True, right_index=True, how="left"
    )
    tessellation_id = cells_m[id_name]

    return blocks, tessellation_id


def buffered_limit(
    gdf: GeoDataFrame | GeoSeries,
    buffer: float | str = 100,
    min_buffer: float = 0,
    max_buffer: float = 100,
    **kwargs,
) -> shapely.Geometry:
    """
    Define limit for tessellation as a buffer around buildings.

    The function calculates a buffer around buildings and returns a MultiPolygon or
    Polygon defining the study area. The buffer can be either a fixed number or
    "adaptive" which calculates the buffer based on Gabriel graph.

    See :cite:`fleischmann2020` for details.

    Parameters
    ----------
    gdf : GeoDataFrame | GeoSeries
        A GeoDataFrame containing building footprints.
    buffer : float | str, optional
        A buffer around buildings limiting the extend of tessellation. If "adaptive",
        the buffer is calculated based on Gabriel graph as the half of the maximum
        distance between neighbors (represented as centroids) of each node + 10% of
        such the maximum distance. The lower and upper bounds can be furhter specified
        by ``min_buffer`` and ``max_buffer``. By default 100.
    min_buffer : float, optional
        The minimum adaptive buffer distance. By default 0.
    max_buffer : float, optional
        The maximum adaptive buffer distance. By default 100.
    **kwargs
        Keyword arguments passed to :meth:`geopandas.GeoSeries.buffer`.

    Returns
    -------
    MultiPolygon
        A MultiPolygon or Polygon defining the study area.

    Examples
    --------
    >>> path = momepy.datasets.get_path("bubenec")
    >>> buildings = geopandas.read_file(path, layer="buildings")
    >>> buildings.head()
       uID                                           geometry
    0    1  POLYGON ((1603599.221 6464369.816, 1603602.984...
    1    2  POLYGON ((1603042.88 6464261.498, 1603038.961 ...
    2    3  POLYGON ((1603044.65 6464178.035, 1603049.192 ...
    3    4  POLYGON ((1603036.557 6464141.467, 1603036.969...
    4    5  POLYGON ((1603082.387 6464142.022, 1603081.574...

    >>> limit = momepy.buffered_limit(buildings)
    >>> type(limit)
    <class 'shapely.geometry.polygon.Polygon'>
    """
    if buffer == "adaptive":
        gabriel = Graph.build_triangulation(gdf.centroid, "gabriel", kernel="identity")
        max_dist = gabriel.aggregate("max")
        buffer = np.clip(max_dist / 2 + max_dist * 0.1, min_buffer, max_buffer).values

    elif not isinstance(buffer, int | float):
        raise ValueError("`buffer` must be either 'adaptive' or a number.")

    return (
        gdf.buffer(buffer, **kwargs).union_all()
        if GPD_GE_10
        else gdf.buffer(buffer, **kwargs).unary_union
    )


def get_network_ratio(df, edges, initial_buffer=500):
    """
    Link polygons to network edges based on the proportion of overlap (if a cell
    intersects more than one edge). Useful if you need to link enclosed tessellation to
    street network. Ratios can be used as weights when linking network-based values
    to cells. For a purely distance-based link use :func:`momepy.get_nearest_street`.
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
    momepy.get_node_id

    Examples
    --------
    >>> links = mm.get_network_ratio(enclosed_tessellation, streets)  # doctest: +SKIP
    >>> links.head()  # doctest: +SKIP
      edgeID_keys                              edgeID_values
    0        [34]                                      [1.0]
    1     [0, 34]  [0.38508998545027145, 0.6149100145497285]
    2        [32]                                        [1]
    3         [0]                                      [1.0]
    4        [26]                                        [1]
    """

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
    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])  # doctest: +SKIP
    """  # noqa
    if limit is not None:
        if isinstance(limit, BaseGeometry):
            limit = gpd.GeoSeries([limit], crs=primary_barriers.crs)
        if limit.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            limit_b = limit.boundary
        else:
            limit_b = limit
        barriers = pd.concat([primary_barriers.geometry, limit_b.geometry])
    else:
        barriers = primary_barriers
    unioned = barriers.union_all() if GPD_GE_10 else barriers.unary_union
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=primary_barriers.crs)

    if additional_barriers is not None:
        if not isinstance(additional_barriers, list):
            raise TypeError(
                "`additional_barriers` expects a list of GeoDataFrames "
                f"or GeoSeries. Got {type(additional_barriers)}."
            )
        additional = pd.concat([gdf.geometry for gdf in additional_barriers])

        inp, res = enclosures.sindex.query(additional.geometry, predicate="intersects")
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

        final_enclosures = pd.concat(
            [
                gpd.GeoSeries(enclosures).drop(unique),
                gpd.GeoSeries(new, crs=primary_barriers.crs),
            ]
        ).reset_index(drop=True)

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
        _, encl_index = final_enclosures.representative_point().sindex.query(
            limit.geometry, predicate="contains"
        )
        keep = np.unique(encl_index)
        return final_enclosures.iloc[keep]

    return final_enclosures
