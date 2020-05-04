Using spatial weights matrix
============================

To capture patterns in urban form, we have to analyze morphometric
characters within a spatial context, taking into the account
neighbouring elements. A simple example could be a mean height within
100 metres (measured as a location-based character for each building).
``momepy`` is using ``libpysal`` spatial weights to capture the
relationship between elements, in this case, buildings, in the form of a
binary matrix (1 = neighbours, 0 = not neighbours). Spatial weights are
an essential part of ``momepy`` and are used on many occasions and in
various forms. However, they should always be based on the unique ID of
element, not index (there are a few exceptions documented in API).

This section covers: 

.. toctree::
   :maxdepth: 2

   weights_nb
   examples
   diversity
   two
