Network analysis
=================

Part of the morphometric analysis is an analysis of street networks.
While ``momepy`` at the moment focuses more on smaller-scale analysis,
some of the network-based characters are included. The main difference
here is that functions in the ``graph`` module, allowing this kind of
study, are based on ``networkx.Graph``, not ``geopandas.GeoDataFrame``.

This section covers: 

.. toctree::
   :maxdepth: 2

   convert
   network
   centrality
   coins
