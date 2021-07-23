.. _api_ref:

.. automodule:: momepy

.. currentmodule:: momepy


momepy API reference
======================

elements
--------

.. autosummary::
   :toctree: generated/

   Blocks
   buffered_limit
   COINS
   enclosures
   get_network_id
   get_network_ratio
   get_node_id
   Tessellation

dimension
---------

.. autosummary::
   :toctree: generated/

   Area
   AverageCharacter
   CourtyardArea
   CoveredArea
   FloorArea
   LongestAxisLength
   Perimeter
   PerimeterWall
   SegmentsLength
   StreetProfile
   Volume
   WeightedCharacter

shape
-----

.. autosummary::
  :toctree: generated/

  CentroidCorners
  CircularCompactness
  CompactnessWeightedAxis
  Convexity
  Corners
  CourtyardIndex
  Elongation
  EquivalentRectangularIndex
  FormFactor
  FractalDimension
  Linearity
  Rectangularity
  ShapeIndex
  SquareCompactness
  Squareness
  VolumeFacadeRatio

spatial distribution
--------------------
.. autosummary::
  :toctree: generated/

  Alignment
  BuildingAdjacency
  CellAlignment
  MeanInterbuildingDistance
  NeighborDistance
  NeighboringStreetOrientationDeviation
  Neighbors
  Orientation
  SharedWalls
  SharedWallsRatio
  StreetAlignment

intensity
---------
.. autosummary::
   :toctree: generated/

   AreaRatio
   BlocksCount
   Count
   Courtyards
   Density
   NodeDensity
   Reached

graph
------------
.. autosummary::
   :toctree: generated/

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
   proportion
   straightness_centrality
   subgraph

diversity
---------
.. autosummary::
   :toctree: generated/

   Gini
   Percentiles
   Range
   Shannon
   Simpson
   Theil
   Unique

   shannon_diversity
   simpson_diversity

spatial weights
---------------
.. autosummary::
   :toctree: generated/

   DistanceBand
   sw_high

preprocessing
-------------
.. autosummary::
   :toctree: generated/

   close_gaps
   CheckTessellationInput
   extend_lines
   remove_false_nodes
   preprocess


utilities
---------
.. autosummary::
   :toctree: generated/

   gdf_to_nx
   limit_range
   nx_to_gdf
   unique_id
