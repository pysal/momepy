.. _api_ref:

.. automodule:: momepy

.. currentmodule:: momepy


momepy API reference
======================

elements
--------

.. autosummary::
   :toctree: generated/

   blocks
   get_network_id
   tessellation

dimension
---------

.. autosummary::
   :toctree: generated/

   area
   courtyard_area
   floor_area
   longest_axis_length
   mean_character
   perimeter
   street_profile
   volume
   weighted_character

shape
-----

.. autosummary::
  :toctree: generated/

  centroid_corners
  circular_compactness
  compactness_weighted_axis
  convexeity
  corners
  courtyard_index
  elongation
  equivalent_rectangular_index
  form_factor
  fractal_dimension
  linearity
  rectangularity
  shape_index
  square_compactness
  squareness
  volume_facade_ratio

spatial distribution
--------------------
.. autosummary::
  :toctree: generated/

  alignment
  building_adjacency
  cell_alignment
  mean_interbuilding_distance
  neighbour_distance
  neighbouring_street_orientation_deviation
  neighbours
  orientation
  shared_walls_ratio
  street_alignment

intensity
---------
.. autosummary::
   :toctree: generated/

   blocks_count
   courtyards
   covered_area_ratio
   elements_in_block
   floor_area_ratio
   frequency
   gross_density

utilities
---------
.. autosummary::
   :toctree: generated/

   Queen_higher
   gdf_to_nx
   nx_to_gdf
   unique_id
