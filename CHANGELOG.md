Changes
=======


Version 0.1.1 (December 15, 2019)
---------------------------------

Small bug-fix release:

- fix for ``AreaRatio`` resetting index of resulting Series (#127)
- fix for ``network_false_nodes`` to work with GeoSeries (#128)
- fix for incomplete spatial_weights and missing neighbors. Instead of raising KeyError
momepy now returns np.nan for affected row. np.nan is also returned if there are no
neighbors (instead of 0). (#131)
