from itertools import combinations, product

import networkx as nx
import numpy as np

from .utils import gdf_to_nx

__all__ = [
    "coins_to_nx",
    "stroke_connectivity",
    "stroke_access",
    "stroke_orthogonality",
    "stroke_spacing",
]


def _get_interior_angle(a, b):
    """
    Helper function for ``make_stroke_graph()``.
    Computes interior angle between two LineString segments
    (interpreted as 2-dimensional vectors)

    Parameters
    ----------
    a, b: numpy.ndarray
    """
    angle = np.rad2deg(
        np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    )
    if angle > 90:
        angle = 180 - angle
    return angle


def _get_end_segment(linestring, point):
    """
    Helper function for ``make_stroke_graph()``.
    Returns the first or last two-Point segment of a LineString.

    Parameters
    ----------
    linestring: shapely.LineString
    point: list
        A list of length 2 containing the coordinates of either
        the first or the last point on the linestring.
    """
    point = tuple(point)
    coords = list(linestring.coords)
    if point == coords[0]:
        geom = [np.array(val) for val in linestring.coords[:2]]
    elif point == coords[-1]:
        geom = [np.array(val) for val in linestring.coords[-2:]]
    else:
        raise ValueError("point is not an endpoint of linestring!")
    return np.array(geom[0] - geom[1])


def coins_to_nx(coins):
    """
    Creates the continuity stroke graph of a street network. The continuity graph
    is similar to, but not identical with, the dual graph. In the continuity graph,
    each stroke (see ``momepy.COINS``) is a node; and each intersection between
    two strokes is an edge.

    Parameters
    ----------
    coins: momepy.COINS
        Continuity strokes computed from a street network.

    Returns
    -------
    stroke_graph : Graph
        A networkx.Graph object.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file(momepy.datasets.get_path("bubenec"), layer="streets")
    >>> coins = momepy.COINS(gdf)
    >>> stroke_graph = momepy.coins_to_nx(coins)
    """

    # get strokes attributes from coins
    stroke_attribute = coins.stroke_attribute()

    # get strokes gdf from coins
    stroke_gdf = coins.stroke_gdf()

    # add stroke_id column
    stroke_gdf["stroke_id"] = stroke_gdf.index

    # add column containing indices of edges comprising each stroke
    # (using COINS.stroke_attribute to map into ID defined in lines gdf)
    stroke_gdf["edge_indices"] = stroke_attribute.groupby(stroke_attribute).apply(
        lambda group: group.index.tolist()
    )

    # recreate primal graph from coins.edge_gdf
    edge_gdf = coins.edge_gdf
    # Add stroke ID to each edge
    edge_gdf["stroke_id"] = coins.stroke_attribute()
    graph = gdf_to_nx(edge_gdf, preserve_index=True, approach="primal")

    # make stroke graph
    stroke_graph = nx.Graph()

    # copy crs and approach attributes from "original" primal graph
    stroke_graph.graph["crs"] = graph.graph["crs"]
    stroke_graph.graph["approach"] = "continuity"

    # add nodes to stroke graph
    stroke_graph.add_nodes_from(
        [
            (
                row.stroke_id,
                {
                    "edge_indices": row.edge_indices,
                    "geometry": row.geometry,  # "geometry" is the representative point
                    "stroke_length": row.geometry.length,
                    "connectivity": 0,
                },
            )
            for _, row in stroke_gdf.iterrows()
        ]
    )

    # add edges to stroke graph
    for n in graph.nodes:
        strokes_present = [
            graph.edges[e]["stroke_id"] for e in graph.edges(n, keys=True)
        ]
        # If strokes intersecting, add the edge if not already present
        if len(set(strokes_present)) > 1:
            for u, v in combinations(set(strokes_present), 2):
                # Find all edges touching the node for both strokes checked
                edges_u = [
                    e
                    for e in graph.edges(n, keys=True)
                    if graph.edges[e]["stroke_id"] == u
                ]
                edges_v = [
                    e
                    for e in graph.edges(n, keys=True)
                    if graph.edges[e]["stroke_id"] == v
                ]
                angle_list = []
                angle_dict = {}
                # Choose the smallest list as number of angles kept
                chosen, other = sorted([edges_u, edges_v], key=len)
                # Find the angles
                for ce, oe in list(product(chosen, other)):
                    point = [graph.nodes[n]["x"], graph.nodes[n]["y"]]
                    gc = _get_end_segment(graph.edges[ce]["geometry"], point)
                    go = _get_end_segment(graph.edges[oe]["geometry"], point)
                    if ce in angle_dict:
                        angle_dict[ce].append(_get_interior_angle(gc, go))
                    else:
                        angle_dict[ce] = [_get_interior_angle(gc, go)]
                # Keep the smallest angles
                angle_list = [min(angle_dict[ekey]) for ekey in angle_dict]
                if stroke_graph.has_edge(u, v):
                    stroke_graph.edges[u, v]["angles"] += angle_list
                    stroke_graph.edges[u, v]["number_connections"] = len(
                        stroke_graph.edges[u, v]["angles"]
                    )
                else:
                    stroke_graph.add_edge(
                        u,
                        v,
                        angles=angle_list,
                        number_connections=len(angle_list),
                    )

    return stroke_graph


def stroke_connectivity(stroke_graph):
    """
    Computes the stroke's connectivity. Connectivity is defined as
    the number of street segments a stroke intersects. Comparing to the degree,
    the same stroke can intersects via several street segments with another stroke.

    Adapted from :cite:`el2022urban`.

    Parameters
    ----------
    stroke_graph: nx.Graph()
        Stroke graph of a network, generated with momepy.coins_to_nx().

    Returns
    ----------
    stroke_graph: nx.Graph()
        Returns stroke_graph where each node has acquired the additional
        attribute `stroke_connectivity`.

    Examples
    --------
    >>> stroke_graph = stroke_connectivity(stroke_graph) # doctest: +SKIP
    """

    for n in stroke_graph.nodes:
        stroke_graph.nodes[n]["stroke_connectivity"] = sum(
            [stroke_graph.edges[e]["number_connections"] for e in stroke_graph.edges(n)]
        )

    return stroke_graph


def stroke_access(stroke_graph):
    """
    Computes the stroke's access. Access is defined as the difference
    between the degree and the connectivity of a stroke. See
    :func:`compute_stroke_connectivity` for a definition of connectivity.

    Adapted from :cite:`el2022urban`.

    Parameters
    ----------
    stroke_graph: nx.Graph()
        Stroke graph of a network, generated with momepy.coins_to_nx().

    Returns
    ----------
    stroke_graph: nx.Graph()
        Returns stroke_graph where each node has acquired the additional
        attribute `stroke_access`; and the additional attribute(s)
        `stroke_connectivity` and `stroke_degree`
        (unless they have been present in the input graph).

    Examples
    --------
    >>> stroke_graph = stroke_access(stroke_graph) # doctest: +SKIP
    """

    # if it doesn't exist as attribute yet, add stroke connectivity
    if not bool(nx.get_node_attributes(stroke_graph, "stroke_connectivity")):
        stroke_graph = stroke_connectivity(stroke_graph)

    # if it doesn't exist as attribute yet, add stroke degree
    if not bool(nx.get_node_attributes(stroke_graph, "stroke_degree")):
        nx.set_node_attributes(
            stroke_graph, dict(nx.degree(stroke_graph)), "stroke_degree"
        )

    # add stroke access (computed via stroke connectivity and stroke degree)
    for n in stroke_graph.nodes:
        stroke_graph.nodes[n]["stroke_access"] = (
            stroke_graph.nodes[n]["stroke_connectivity"]
            - stroke_graph.nodes[n]["stroke_degree"]
        )

    return stroke_graph


def stroke_orthogonality(stroke_graph):
    """
    Computes the stroke's orthogonality. Orthogonality is defined
    as the average of the sine of the minimum angles between the
    stroke and the street segments it intersects:

    .. math::
        O(s)=\\frac{\\sum_{i\\in A}sin(\\theta_i)}{C(s)}

    Where :math:`\\theta_i` is the minimum angle between the street segment
    :math:`i` and the stroke :math:`s`, and :math:`C(s)` is the
    connectivity of the stroke :math:`s`.

    Its value vary between 0 and 1, for low to right angles.

    Adapted from :cite:`el2022urban`.

    Parameters
    ----------
    stroke_graph: nx.Graph()
        Stroke graph of a network, generated with momepy.coins_to_nx().

    Returns
    ----------
    stroke_graph: nx.Graph()
        Returns stroke_graph where each node has acquired the additional
        attribute `stroke_orthogonality`; and the additional attribute
        `stroke_connectivity` (unless it has been present in the input graph).

    Examples
    --------
    >>> stroke_graph = stroke_orthogonality(stroke_graph) # doctest: +SKIP
    """

    # if it doesn't exist as attribute yet, add stroke connectivity
    if not bool(nx.get_node_attributes(stroke_graph, "stroke_connectivity")):
        stroke_graph = stroke_connectivity(stroke_graph)

    for n in stroke_graph.nodes:
        # get angles for that stroke
        angles = [
            val
            for e in stroke_graph.edges(n)
            if stroke_graph.edges[e]["angles"]
            for val in stroke_graph.edges[e]["angles"]
        ]
        # get orthogonality
        stroke_graph.nodes[n]["stroke_orthogonality"] = (
            sum([np.sin(np.deg2rad(angle)) for angle in angles])
            / stroke_graph.nodes[n]["stroke_connectivity"]
        )

    return stroke_graph


def stroke_spacing(stroke_graph):
    """
    Computes the stroke's spacing. Spacing is defined as the
    average lenght between connections for a stroke.

    Adapted from :cite:`el2022urban`.

    Parameters
    ----------
    stroke_graph: nx.Graph()
        Stroke graph of a network, generated with momepy.coins_to_nx().

    Returns
    ----------
    stroke_graph: nx.Graph()
        Returns stroke_graph where each node has acquired the additional
        attribute `stroke_spacing`; and the additional attribute
        `stroke_connectivity` (unless it has been present in the input graph).

    Examples
    --------
    >>> stroke_graph = stroke_spacing(stroke_graph) # doctest: +SKIP
    """

    # if it doesn't exist as attribute yet, add stroke connectivity
    if not bool(nx.get_node_attributes(stroke_graph, "stroke_connectivity")):
        stroke_graph = stroke_connectivity(stroke_graph)

    for n in stroke_graph.nodes:
        stroke_graph.nodes[n]["stroke_spacing"] = (
            stroke_graph.nodes[n]["stroke_length"]
            / stroke_graph.nodes[n]["stroke_connectivity"]
        )

    return stroke_graph
