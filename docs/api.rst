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
   get_network_id
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
  Convexeity
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
   clustering
   cyclomatic
   edge_node_ratio
   gamma
   global_closeness_centrality
   local_closeness_centrality
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

   Range
   Simpson
   Theil
   Gini

utilities
---------
.. autosummary::
   :toctree: generated/

   gdf_to_nx
   limit_range
   network_false_nodes
   nx_to_gdf
   preprocess
   snap_street_network_edge
   sw_high
   unique_id
