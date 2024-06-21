.. _api_ref:

.. automodule:: momepy

.. currentmodule:: momepy

momepy API reference
======================

The current version of momepy includes two implementations of most of the functionality
due to the ongoing migration period moving from the legacy class-based API to a new
function-based API. This page outlines the stable API. For the legacy functionality, see
:doc:`Legacy API  <legacy_api>`.

Managing morphological elements
-------------------------------
.. _elements:

Momepy allows creation of a small subset of bespoke morphological geometric features.

.. autosummary::
   :toctree: api/

   morphological_tessellation
   enclosed_tessellation
   enclosures
   generate_blocks

Additionally, it contains tools supporting these.

.. autosummary::
   :toctree: api/

   buffered_limit
   verify_tessellation

And tools linking various elements together.

.. autosummary::
   :toctree: api/

   get_nearest_street
   get_nearest_node
   get_network_ratio

Measuring dimension
-------------------

A set of functions to measure dimensions of geometric elements:

.. autosummary::
   :toctree: api/

   courtyard_area
   floor_area
   longest_axis_length
   perimeter_wall
   street_profile
   volume
   weighted_character

Measuring shape
---------------

A set of functions to measure shape of geometric elements:

.. autosummary::
   :toctree: api/

   centroid_corner_distance
   circular_compactness
   compactness_weighted_axis
   convexity
   corners
   courtyard_index
   elongation
   equivalent_rectangular_index
   facade_ratio
   form_factor
   fractal_dimension
   linearity
   rectangularity
   shape_index
   square_compactness
   squareness

Measuring spatial distribution
------------------------------

A set of functions to measure spatial distribution of geometric elements:

.. autosummary::
   :toctree: api/

   alignment
   building_adjacency
   cell_alignment
   mean_interbuilding_distance
   neighbor_distance
   neighbors
   orientation
   shared_walls
   street_alignment

Measuring intensity
-------------------

A set of functions to measure intensity characters:

.. autosummary::
   :toctree: api/

   courtyards

Note that additional intensity characters can be directly derived using :meth:`libpysal.graph.Graph.describe`
and functions :func:`describe_agg` and :func:`describe_reached_agg`.


Measuring diversity
-------------------

A set of functions to measure spatial diversity of elements and their values:

.. autosummary::
   :toctree: api/

   describe_agg
   describe_reached_agg
   gini
   percentile
   shannon
   simpson
   theil
   values_range
   mean_deviation

Note that additional diversity characters can be directly derived using :meth:`libpysal.graph.Graph.describe`.

Underlying components of :func:`shannon` and :func:`simpson` are also exposed for direct use:


.. autosummary::
   :toctree: api/

   shannon_diversity
   simpson_diversity

Measuring connectivity
----------------------

A set of functions for the analysis of connectivity and configuration of street networks:

.. autosummary::
   :toctree: api/

   betweenness_centrality
   cds_length
   closeness_centrality
   clustering
   cyclomatic
   edge_node_ratio
   gamma
   mean_node_degree
   mean_node_dist
   mean_nodes
   meshedness
   node_degree
   node_density
   proportion
   straightness_centrality
   subgraph
   COINS

With utilities allowing conversion between networkx objects and GeoPandas objects.

.. autosummary::
   :toctree: api/

   gdf_to_nx
   nx_to_gdf

Data preprocessing
------------------

Most of the algorithms have certain expectations about the quality of input data. The
`preprocessing` module helps adapting the input data and fixing common issues.

.. autosummary::
   :toctree: api/

   close_gaps
   extend_lines
   remove_false_nodes
   consolidate_intersections
   roundabout_simplification

Additionally, there are methods for data assessment.

.. autosummary::
   :toctree: api/

   CheckTessellationInput
   FaceArtifacts


Further analysis can be done directly using methods available in :class:`libpysal.graph.Graph`.

.. toctree::
   :maxdepth: 1
   :hidden:

   legacy_api