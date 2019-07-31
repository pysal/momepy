Install
=======

Momepy is not yet released. At this moment, you can either download the GitHub
repository to work with ``momepy`` or install the latest development version
from GitHub via PIP::

    pip install git+git://github.com/martinfleis/momepy.git

Please ensure that you have all dependencies installed as ``momepy`` setup
currently does not specify dependency requirements.
Momepy depends on the following packages: ``geopandas``, ``libpysal``
and ``tqdm``. All of the packages are currently required, which will be
changed in future.

Install via Conda
-----------------

As `momepy` is dependent on `geopandas` and other spatial packages, we recommend
to install all dependencies via `conda` from `conda-forge`::

    conda config --add channels conda-forge  # ensure that conda-forge is in your list of channels on the top
    conda config --set channel_priority strict  # strict priority to install all from conda-forge
    conda create --name momepy_env geopandas libpysal tqdm
    conda activate momepy_env
    pip install git+git://github.com/martinfleis/momepy.git
