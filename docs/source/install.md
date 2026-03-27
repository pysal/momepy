# Install

Momepy, similar to GeoPandas, can be a bit complicated to install. However, if
you follow the recommended instructions below, there should be no issue. For
more details on issues with the geospatial Python stack, please refer to the
[GeoPandas installation instructions](http://geopandas.org/install.html).

## Install via Conda

As `momepy` is dependent on [geopandas](https://geopandas.org/) and other
spatial packages, we recommend installing all dependencies via
[conda](https://conda.io/en/latest/) from
[conda-forge](https://conda-forge.org/):

```bash
conda install -c conda-forge momepy
```

Conda should be able to resolve any dependency conflicts and install momepy
together with all necessary dependencies.

If you do not have [conda-forge](https://conda-forge.org/) in your conda
channels, you can add it using:

```bash
conda config --add channels conda-forge
```

To ensure that all dependencies will be installed from
[conda-forge](https://conda-forge.org/), we recommend using strict channel
priority:

```bash
conda config --env --set channel_priority strict
```

```{note}
We strongly recommend installing everything from the *conda-forge* channel.
Mixing conda channels, or mixing conda and pip packages, can lead to import
problems.
```

### Creating a New Environment for momepy

If you want to make sure that everything will work as it should, you can create
a new conda environment for momepy. Assuming we want to create a new
environment called `momepy_env`:

```bash
conda create -n momepy_env
conda activate momepy_env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install momepy
```

## Install via pip

Momepy is also available on PyPI, but ensure that all dependencies are properly
installed before installing momepy. Some C dependencies cause problems with
installing using pip only:

```bash
pip install momepy
```

## Install from the Repository

If you want to work with the latest development version of momepy, you can do
so by cloning the [GitHub repository](https://github.com/pysal/momepy) and
installing momepy from the local directory:

```bash
git clone https://github.com/pysal/momepy.git
cd momepy
pip install .
```

Alternatively, you can install the latest version directly from GitHub:

```bash
pip install git+https://github.com/pysal/momepy.git
```

Installing directly from the repository might face the same dependency issues
described above for installing using pip. To ensure that the environment is
properly prepared and every dependency works as intended, you can install them
using conda before installing the development version of momepy:

```bash
conda install -c conda-forge geopandas networkx libpysal tqdm
```

## Dependencies

Required dependencies:

- [geopandas](https://geopandas.org/)
- [libpysal](http://pysal.org/libpysal)
- [networkx](http://networkx.github.io)
- [tqdm](https://tqdm.github.io)

Some functions also depend on additional packages, which are optional:

- [mapclassify](http://pysal.org/mapclassify)
- [inequality](http://pysal.org/inequality)
- [numba](https://numba.pydata.org)
- [esda](http://pysal.org/esda)
