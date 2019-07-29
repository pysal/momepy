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
   buffered_limit
   get_network_id
   get_node_id
   tessellation

dimension
---------

.. autosummary::
   :toctree: generated/

   area
   courtyard_area
   covered_area
   floor_area
   longest_axis_length
   mean_character
   perimeter
   segments_length
   street_profile
   volume
   wall
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
   density
   elements_count
   floor_area_ratio
   node_density
   reached

graph
------------
.. autosummary::
   :toctree: generated/

   cds_length
   clustering
   cyclomatic
   edge_node_ratio
   eigenvector
   gamma
   local_closeness
   mean_node_degree
   mean_node_dist
   meshedness
   node_degree
   proportion

diversity
---------
.. autosummary::
   :toctree: generated/

   rng
   simpson
   theil

utilities
---------
.. autosummary::
   :toctree: generated/

   gdf_to_nx
   limit_range
   network_false_nodes
   nx_to_gdf
   preprocess
   unique_id
   Queen_higher
