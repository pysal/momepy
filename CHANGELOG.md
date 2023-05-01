Changelog
=========

Version 0.6.0 (May 1, 2023)
---------------------------

Momepy now requires shapely 2.0 or newer to run and no longer depends on PyGEOS. The
highlights of the release:

Enhancements:

- ENH: support bi-directional roads in ``gdf_to_nx`` (#357)
- ENH: geometry-based simplification of roundabouts (#371, #392)
- REF: update for shapely 2.0 (#479)

Bug fixes:

- resolves bug in ``Percentiles`` append. (#427)

Version 0.5.4 (September 20, 2022)
----------------------------------

Minor patch bringing some bug fixes and ensuring the full compatibility with the recent
versions of dependencies.

Fixes:

- BUG: return itself when no change possible in remove_false_nodes (#365)
- BUG: make COINS independent of the GeoDataFrame index (#378)
- Added warning for gdp_to_nx if geometries are not LineStrings (#379)
- REF: remove deprecated pandas append (#382, #384)
- MAINT: address numpy.nanpercintle warning (#387)
- MAINT: handle scenarios leading to dimensions.StreetProfile() warnings (#389)
- handle All NaN slice warning (#395)
- Fix various warnings (#398)
- BUG: get_network_ratio non-interescting joins fix (#409)

Version 0.5.3 (April 9, 2022)
-----------------------------

Minor patch release primarily fixing an issue with momepy.Blocks and a creation of angular graphs.

Fixes:

- BUG: Fix angle computation in graph creation with dual approach (#347)
- BUG: fix issue with blocks within another blocks (#351)

Version 0.5.2 (January 6, 2022)
-------------------------------

Since version 0.5.2, momepy is licensed under the BSD-3-Clause License instead
of the MIT License, to be in line with the rest of the PySAL ecosystem.

Fixes:

- BUG: non-default index dropped in Blocks id series (#315)
- BUG: fix FormFactor formula (#330)

Version 0.5.1 (October 20, 2021)
--------------------------------

Small patch adjusting momepy to changes geopandas 0.10.

Fixes:

- BUG: non-default index dropped in Blocks id series
- REF/TST: minimise warnings from geopandas 0.10

Version 0.5.0 (September 12, 2021)
----------------------------------

Enhancements:

- COINS algorithm for analysis of continuity of street networks (#248, #276)
- ENH: distance decay in Percentiles (#269)
- ENH: add dropna keyword to Unique (#280)
- ENH: catch geographic CRS in Tessellation (#298)
- ENH: support shapely polygon as enclosures limit (#299)

Bug fixes:

- BUG/DOC: adapt to mutliindex + remove preprocess (#281)
- BUG: Tessellation error on non-standard enclosures (#291)
- BUG: properly clip enclosures by limit (#288)

Other:

- DEP: remove deprecated functions and args (#268)
- PERF: use dask.bag in Tessellation (#296)

Version 0.4.4 (April 30, 2021)
---------------------------------

Small bug fix affecting `nx_to_gdf` and `NodeDensity`.

- BUG: node ID doesn't match ID in weights (#274)

Version 0.4.3 (February 16, 2021)
---------------------------------

Small bug fix:

- BUG: UnboundLocalError: local variable 'cpu_count' referenced before assignment (#266)

Version 0.4.2 (February 4, 2021)
--------------------------------

Bug fix release:

- BUG: resolve nans in StreetProfile (#251)
- REGR: fix slowdown in tessellation (#256)
- BUG: avoid string "geometry" (#257)
- PERF: move translation in Tessellation (#259)
- REGR: use convex_hull to mitigate infinity (#260)

Version 0.4.1 (January 12, 2021)
--------------------------------

- fixed bug in the automatic selection of number of dask chunks in enclosed tessellation
- fixed infinity issue in ``StreetProfile`` (#249)
- fixed issue changing the original dataframe in ``DistanceBand`` (#250)

Version 0.4.0 (December 26, 2020)
---------------------------------

Requirements:

- momepy now requires GeoPandas 0.8 or newer
- momepy now requires pygeos
- momepy now (optionally) requires mapclassify 2.4.2 or newer

API changes:

- ``network_false_nodes`` is now deprecated. Use new ``remove_false_nodes`` instead.

Enhancements:

- New performant algorithm ``remove_false_nodes`` to remove nodes of a degree 2 of a LineString network. (#204)
- Faster ``CircularCompactness`` (#205)
- pygeos-based ``Tessellation`` (#207)
- New class ``Percentiles`` (#209)
- Various speedups (#209)
- New ``enclosures`` function (#211)
- Enclosed tessellation option in ``Tessellation`` (#212)
- Preprocessing module (#214)
- Preprocessing function to ``close_gaps`` of LineString geoemtry (#215)
- Preprocessing function to ``extend_lines`` (#217)
- ratio-based network links (#218)
- vectorize ``StreetProfile`` (#219)
- capture MutliLineString in ``Linearity`` (#236)
- support MultiPolygons (#234)
- handle NaNs in ``limit_range`` (#235)
- ``SharedWalls`` length (#238)
- refactor ``Blocks`` using overlay and libpysal (#237)
- more options in converting to networkx graphs in ``gdf_to_nx`` (#240)
- use mapclassify.classify in ``Simpson`` and ``Shannon`` (#241)

Bug fixes:

- fix nearest neighbor in ``get_network_ratio`` (#224)
- ``Tessellation`` error when geom collapsed (#226)
- ``Blocks`` empty difference (#230)

Version 0.3.0 (July 29, 2020)
-----------------------------

API changes:

- ``Convexeity`` is now ``Convexity`` (#171)
- ``local_`` centrality (betweenness, closeness, straightness) has been included in respective global versions (#178)

New features:

- ``CheckTessellationInput`` to check building footprint data for potential issues during ``Tessellation`` (#163)
- On demand ``DistanceBand`` spatial weights (neighbors) for larger weight which would not fit in memory (#165)

Enhancements:

- New documentation (#167)
- Support network analysis for full network (#176)
- Options for ``preprocess`` (#180)
- Expose underlying ``simpson`` and ``shannon`` functions (#183)
- ``MeanInterbuildingDistance`` performance (#187)
- ``StreetProfile`` performance refactor (#186)
- Retain attributes in ``network_false_nodes`` (#189)
- Performance improvements in ``elements`` module (#190)
- Performance refactor of ``SharedWallsRatio`` (#191)
- Performance refactor of ``NeighboringStreetOrientationDeviation`` (#192)
- Minor performance improvements (#193, #196)
- Allow specification of verbosity (#195)
- Perfomance enhancements in ``sw_high`` (#198)

Bug fixes:

- ``Density`` TypeError for islands (#164)
- Preserve CRS in ``network_false_nodes`` (#181)
- Fixed ``Squareness`` for non-Polygon geom types (#182)
- CRS lost with older geopandas (#188)

Version 0.2.1 (April 15, 2020)
------------------------------

- fixed regression causing ``MeanInterbuildingDistance`` failure (#161)

Version 0.2.0 (April 13, 2020)
------------------------------

API changes:

- ``AverageCharacter`` allows calculation of multiple modes (mean, median, mode) at the same time. Each can be accessed via its own attribute. Apart from ``mean``, none is accessible using ``.series`` anymore. (#147)

Enhancements:

- ``Shannon`` index (#158)
- ``Simpson`` allows Gini-Simpson and Inverse Simpson index modes (#157)
- Diversity classes support categorical values (#159)
- ``SegmentsLength`` allows sum and mean at the same time (#146)
- ``AverageCharacter`` allows calculation of multiple modes (mean, median, mode) at the same time (#147)
- Better compatibility with OSMnx Graphs (#149)
- ``Orientation`` support LineString geometry (#156)
- ``AreaRatio`` for uneven number of features (#135)
- Performance improvements (#144, #145, #152, #155)

Bug fixes:

- float precision errors in ``network_false_nodes`` (#133)
- ``network_false_nodes`` for multiindex (#136)
- ``BlocksCount`` no neighbors error (#139, #140)
- LineString Z support in ``nx_to_gdf`` (#148)
- accidental 'rtd' print (#150)
- ``CentroidCorner`` may fail for Polygon Z (#151)

Version 0.1.1 (December 15, 2019)
---------------------------------

Small bug-fix release:

- fix for ``AreaRatio`` resetting index of resulting Series (#127)
- fix for ``network_false_nodes`` to work with GeoSeries (#128)
- fix for incomplete spatial_weights and missing neighbors. Instead of raising KeyError momepy now returns np.nan for affected row. np.nan is also returned if there are no neighbors (instead of 0). (#131)
