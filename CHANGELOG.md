Changes
=======

Version 0.2.1 (April 15, 2020)
------------------------------

- fixed regression causing `MeanInterbuildingDistance` failure (#161)


Version 0.2.0 (April 13, 2020)
------------------------------

API changes:

- `AverageCharacter` allows calculation of multiple modes (mean, median, mode) at the same time. Each can be accessed via its own attribute. Apart from `mean`, none is accessible using `.series` anymore. (#147)

Enhancements:

- `Shannon` index (#158)
- `Simpson` allows Gini-Simpson and Inverse Simpson index modes (#157)
- Diversity classes support categorical values (#159)
- `SegmentsLength` allows sum and mean at the same time (#146)
- `AverageCharacter` allows calculation of multiple modes (mean, median, mode) at the same time (#147)
- Better compatibility with OSMnx Graphs (#149)
- `Orientation` support LineString geometry (#156)
- `AreaRatio` for uneven number of features (#135)
- Performance improvements (#144, #145, #152, #155)

Bug fixes:

- float precision errors in `network_false_nodes` (#133)
- `network_false_nodes` for multiindex (#136)
- `BlocksCount` no neighbors error (#139, #140)
- LineString Z support in `nx_to_gdf` (#148)
- accidental 'rtd' print (#150)
- `CentroidCorner` may fail for Polygon Z (#151)


Version 0.1.1 (December 15, 2019)
---------------------------------

Small bug-fix release:

- fix for ``AreaRatio`` resetting index of resulting Series (#127)
- fix for ``network_false_nodes`` to work with GeoSeries (#128)
- fix for incomplete spatial_weights and missing neighbors. Instead of raising KeyError
momepy now returns np.nan for affected row. np.nan is also returned if there are no
neighbors (instead of 0). (#131)
