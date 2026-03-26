Data Structure
=========================================

Momepy is built on top of `geopandas`_ ``GeoDataFrame`` objects and, for
network analysis, on `networkx`_ ``Graph``.

For any kind of morphometric analysis, data needs to be provided as
``GeoDataFrames``. Results of morphometric analysis from ``momepy`` can
be generally returned as pandas ``Series`` to be added as a column of
existing ``GeoDataFrame``. All the detailes and attributes of each class
are clearly described in the :doc:`API <../api>`.

Morphometric functions
----------------------

Morphometric functions available in ``momepy`` could be divided into
four different groups based on their approach to data requirements and
outputs.

1. Simple characters

   Simple morphometric characters are using single ``GeoDataFrame`` as a
   source of the data.


2. Relational characters

   Relational characters are based on relations between two or more
   ``GeoDataFrames``. Typical example is ``street_alignment``, which requires
   a) orientation of streets and b) orientation of buildings.


3. Network analysis

   Network analysis characters are based on
   ``networkx.Graph`` and returns ``networkx.Graph`` with additional
   node or edge attributes.

Morphological elements
----------------------

Additional modules (``elements`` and ``utils``) cover functions
generating new morphological elements (like morphological tessellation)
or links between them. For details, please refer to the :doc:`API <../api>`.

The majority of functions used within ``momepy`` are not limited to one type
of morphological element. However, the whole package is built with a
specific set of elements in mind, based on the research done at the
University of Strathclyde by the `Urban Design Studies Unit`_. This is
true especially for morphological tessellation, partitioning of space
based on building footprints. Morphological tessellation can substitute
plots for certain types of analysis and provide additional information,
like the adjacency, for the other. More information on tessellation is
in dedicated :doc:`section <elements/tessellation>` of this guide.

Generally, we can work with any kind of morphological element which fits
the selected function, there is no restriction. Sometimes, where
documentation refers to buildings, other elements like blocks can be
used as well as long as the principle remains the same.

For example, you can use ``momepy`` to do morphometric analysis of:

-  buildings,
-  plots,
-  morphological cells,
-  streets,

   -  profiles,
   -  networks,

-  blocks,

and more.

Links between elements
----------------------

When using more than one morphological element, ``momepy`` needs to
understand what is the relationship between them. For this, it relies on
indices of GeoPandas objects. It is expected, that every building lies on
certain plot or morphological cell, on certain street or within certain
block. To use ``momepy``, each feature of each layer needs its own
unique index. Moreover, each feature also needs to bear ``index``
of related elements. Consider following sample rows of
``buildings_gdf``:

====== =========== ==================
index  block_index network_edge_index
====== =========== ==================
1           143      22
2           143      25
3           144      25
4           144      25
5           144      29
====== =========== ==================

Each building has its own unique ``index``, while more buildings
share ``block_index`` of block they belong to. In this sense, in
``blocks_gdf`` each feature would have its own unique ``index`` used
as a reference for ``buildings_gdf``. In principle, elements on the
smaller scale contains index information of elements on the larger - blocks will not
have building index.

Momepy can :doc:`link certain types of elements together <elements/links>`

Spatial graphs
--------------

Unique indices are also used as an index within spatial graphs.
Thanks to this, spatial graphs generated on morphological tessellation
(like Queen contiguity) can be directly used on buildings and vice
versa. Detailed information on using spatial weights within momepy will
be :doc:`discussed later <weights/weights>`.

.. _link certain types of elements together: elements/links
.. _geopandas: http://geopandas.org
.. _networkx: https://networkx.github.io
.. _Urban Design Studies Unit: http://www.udsu-strath.com
