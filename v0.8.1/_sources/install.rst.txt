Install
=======

Momepy, similar to GeoPandas, can be a bit complicated to install. However,
if you follow recommended instructions below, there should be no issue. For
more details on issues with geospatial python stack, please refer to `GeoPandas
installation instructions <http://geopandas.org/install.html>`__.

Install via Conda
-----------------

As ``momepy`` is dependent on `geopandas`_ and other spatial packages, we recommend
to install all dependencies via `conda`_ from `conda-forge`_::

    conda install -c conda-forge momepy

Conda should be able to resolve any dependency conflicts and install momepy
together with all necessary dependencies.

If you do not have `conda-forge`_ in your conda channels, you can add it using::

    conda config --add channels conda-forge

To ensure that all dependencies will be installed from `conda-forge`_, we recommend
using strict channel priority::

    conda config --env --set channel_priority strict

.. note::

    We strongly recommend to install everything from the *conda-forge* channel.
    Mixture of conda channels or conda and pip packages can lead to import problems.


Creating a new environment for momepy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to make sure, that everything will work as it should, you can create
a new conda environment for momepy. Assuming we want to create a new environment
called ``momepy_env``::

    conda create -n momepy_env
    conda activate momepy_env
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install momepy


Install via pip
---------------

Momepy is also available on PyPI, but ensure that all dependencies are properly
installed before installing momepy. Some C dependencies are causing problems with
installing using pip only::

    pip install momepy


Using Docker
------------

You can also use `Dockerfile <https://github.com/pysal/momepy/tree/main/environments>`_
to build minimal a environment for momepy, or
get the official image from Docker Hub::

    docker pull martinfleis/momepy:<version>

For example::

    docker pull martinfleis/momepy:0.4

Note that images are available from version 0.4.

See more in the
`ReadMe <https://github.com/pysal/momepy/blob/main/environments/Readme.md>`_

If you need the full stack of geospatial Python libraries, use `darribas/gds_env <https://darribas.org/gds_env/>`_
which provides the updated platform for Geographic Data Science (including momepy)::

    docker pull darribas/gds_py


Install from the repository
---------------------------

If you want to work with the latest development version of momepy, you can do so
by cloning `GitHub repository <https://github.com/pysal/momepy>`__ and
installing momepy from local directory::

    git clone https://github.com/pysal/momepy.git
    cd momepy
    pip install .

Alternatively, you can install the latest version directly from GitHub::

    pip install git+git://github.com/pysal/momepy.git

Installing directly from repository might face the same dependency issues as
described above regarding installing using pip. To ensure that environment is
properly prepared and every dependency will work as intended, you can install
them using conda before installing development version of momepy::

    conda install -c conda-forge geopandas networkx libpysal tqdm


Dependencies
------------

Required dependencies:

- `geopandas`_ (>= 0.12.0)
- `libpysal`_ (>= 4.12.0)
- `networkx`_
- `tqdm`_

Some functions also depend on additional packages, which are optional:

- `mapclassify`_ (>= 2.4.2)
- `inequality`_
- `numba`_
- `esda`_


.. _geopandas: https://geopandas.org/

.. _mapclassify: http://pysal.org/mapclassify

.. _esda: http://pysal.org/esda

.. _libpysal: http://pysal.org/libpysal

.. _inequality: http://pysal.org/inequality

.. _networkx: http://networkx.github.io

.. _numba: https://numba.pydata.org

.. _tqdm: http://networkx.github.io

.. _pysal: http://pysal.org

.. _conda-forge: https://conda-forge.org/

.. _conda: https://conda.io/en/latest/

