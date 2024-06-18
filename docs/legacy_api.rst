.. _legacy_api_ref:

.. currentmodule:: momepy


Legacy API reference
====================

.. warning::
    The functionality listed below is part of the legacy API and has been deprecated.
    Each class emits ``FutureWarning`` with a deprecation note. If you'd like to
    silence all of these warnigns, set an environment variable ``ALLOW_LEGACY_MOMEPY``
    to ``"True"``::

      import os

      os.environ["ALLOW_LEGACY_MOMEPY"] = "True"

elements
--------

.. autosummary::
   :toctree: generated/

   Blocks
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

   preprocess


utilities
---------
.. autosummary::
   :toctree: generated/

   limit_range
   unique_id