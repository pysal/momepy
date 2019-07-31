Install
=======

You can install ``momepy`` using pip from PyPI::

  pip install momepy

Momepy depends on the following packages: ``geopandas``, ``libpysal``, ``networkx``
and ``tqdm``. Optional dependencies for specific tasks inlcude ``pysal`` and ``mapclassify``.

Install via Conda
-----------------

As `momepy` is dependent on `geopandas` and other spatial packages, we recommend
to install all dependencies via `conda` from `conda-forge`::

    conda config --add channels conda-forge  # ensure that conda-forge is in your list of channels on the top
    conda config --set channel_priority strict  # strict priority to install all from conda-forge
    conda create --name momepy_env geopandas libpysal networkx tqdm
    conda activate momepy_env
    pip install git+git://github.com/martinfleis/momepy.git


CHANGE TO ENSURE IT IS CONDA-FORGE ENV
