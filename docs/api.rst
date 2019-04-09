.. _api_ref:

.. currentmodule:: momepy


momepy API reference
======================

elements
--------

.. autosummary::
   :toctree: generated/

   elements.blocks
   elements.get_network_id
   elements.tessellation

dimension
---------

.. autosummary::
   :toctree: generated/

   dimension.area
   dimension.courtyard_area
   dimension.effective_mesh
   dimension.floor_area
   dimension.longest_axis_length
   dimension.perimeter
   dimension.street_profile
   dimension.volume
   dimension.weighted_character

shape
-----

.. autosummary::
  :toctree: generated/

  shape.centroid_corners
  shape.circular_compactness
  shape.compactness_weighted_axis
  shape.convexeity
  shape.corners
  shape.courtyard_index
  shape.elongation
  shape.equivalent_rectangular_index
  shape.form_factor
  shape.fractal_dimension
  shape.linearity
  shape.rectangularity
  shape.shape_index
  shape.square_compactness
  shape.squareness
  shape.volume_facade_ratio

spatial distribution
--------------------
.. autosummary::
  :toctree: generated/

  distribution.alignment
  distribution.building_adjacency
  distribution.cell_alignment
  distribution.mean_interbuilding_distance
  distribution.neighbour_distance
  distribution.neighbouring_street_orientation_deviation
  distribution.neighbours
  distribution.orientation
  distribution.shared_walls_ratio
  distribution.street_alignment

intensity
---------
.. autosummary::
   :toctree: generated/

   intensity.blocks_count
   intensity.courtyards
   intensity.covered_area_ratio
   intensity.elements_in_block
   intensity.floor_area_ratio
   intensity.frequency
   intensity.gross_density

utilities
---------
.. autosummary::
   :toctree: generated/

   utils.Queen_higher
   utils.gdf_to_nx
   utils.nx_to_gdf
   utils.unique_id
