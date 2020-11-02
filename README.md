# momepy
[![Documentation Status](https://readthedocs.org/projects/momepy/badge/?version=latest)](http://docs.momepy.org/en/latest/?badge=latest) [![Actions Status](https://github.com/martinfleis/momepy/workflows/Tests/badge.svg)](https://github.com/martinfleis/momepy/actions?query=workflow%3ATests)
[![codecov](https://codecov.io/gh/martinfleis/momepy/branch/master/graph/badge.svg)](https://codecov.io/gh/martinfleis/momepy) [![CodeFactor](https://www.codefactor.io/repository/github/martinfleis/momepy/badge)](https://www.codefactor.io/repository/github/martinfleis/momepy) [![DOI](https://joss.theoj.org/papers/10.21105/joss.01807/status.svg)](https://doi.org/10.21105/joss.01807)


<img src="https://raw.githubusercontent.com/martinfleis/momepy/master/docs/_static/logo.png" width="50%">

## Introduction
Momepy is a library for quantitative analysis of urban form - urban morphometrics. It is built on top of [GeoPandas](http://geopandas.org), [PySAL](http://pysal.org) and [networkX](http://networkx.github.io).

> *momepy* stands for Morphological Measuring in Python

Some of the functionality that momepy offers:

- Measuring [dimensions](https://docs.momepy.org/en/latest/api.html#dimension) of morphological elements, their parts, and aggregated structures.
- Quantifying [shapes](https://docs.momepy.org/en/latest/api.html#shape) of geometries representing a wide range of morphological features.
- Capturing [spatial distribution](https://docs.momepy.org/en/latest/api.html#spatial-distribution) of elements of one kind as well as relationships between different kinds.
- Computing density and other types of [intensity](https://docs.momepy.org/en/latest/api.html#intensity) characters.
- Calculating [diversity](https://docs.momepy.org/en/latest/api.html#diversity) of various aspects of urban form.
- Capturing [connectivity](https://docs.momepy.org/en/latest/api.html#graph) of urban street networks
- Generating relational [elements](https://docs.momepy.org/en/latest/api.html#elements) of urban form (e.g. morphological tessellation)

Momepy aims to provide a wide range of tools for a systematic and exhaustive analysis of urban form. It can work with a wide range of elements, while focused on building footprints and street networks.

Comments, suggestions, feedback, and contributions, as well as bug reports, are very welcome.

## Getting Started
Quick and easy [getting started guide](https://guide.momepy.org/getting_started.html) is part of the [User Guide](https://guide.momepy.org/getting_started.html).


## Documentation
Documentation of `momepy` is available at [docs.momepy.org](https://docs.momepy.org/).

## User Guide
User guide with examples of `momepy` usage is available at [guide.momepy.org](https://guide.momepy.org/).

## Examples

```py
coverage = momepy.AreaRatio(tessellation, buildings, left_areas=tessellation.area,
                            right_areas='area', unique_id='uID')
tessellation['CAR'] = coverage.series
```

![Coverage Area Ratio](https://raw.githubusercontent.com/martinfleis/momepy/master/docs/_static/example1.png)

```py
area_simpson = momepy.Simpson(tessellation, values='area',
                              spatial_weights=sw3,
                              unique_id='uID')
tessellation['area_simpson'] = area_simpson.series
```

![Local Simpson's diversity of area](https://raw.githubusercontent.com/martinfleis/momepy/master/docs/_static/diversity_22_0.png)

```py
G = momepy.straightness_centrality(G)
```

![Straightness centrality](https://raw.githubusercontent.com/martinfleis/momepy/master/docs/_static/centrality_27_0.png)


## How to cite
To cite `momepy` please use following [software paper](https://doi.org/10.21105/joss.01807) published in the JOSS.

Fleischmann, M. (2019) ‘momepy: Urban Morphology Measuring Toolkit’, Journal of Open Source Software, 4(43), p. 1807. doi: 10.21105/joss.01807.

BibTeX:

    @article{fleischmann_2019,
        author={Fleischmann, Martin},
        title={momepy: Urban Morphology Measuring Toolkit},
        journal={Journal of Open Source Software},
        year={2019},
        volume={4},
        number={43},
        pages={1807},
        DOI={10.21105/joss.01807}
    }

## Install
You can install `momepy` using Conda from `conda-forge` (recommended):

    conda install -c conda-forge momepy

or from PyPI using `pip`:

    pip install momepy

See the [installation instructions](http://docs.momepy.org/en/latest/install.html) for detailed instructions.
Momepy depends on python geospatial stack, which might cause some dependency issues.

## Contributing to momepy
Contributions of any kind to momepy are more than welcome. That does not mean new code only, but also improvements of documentation and user guide, additional tests (ideally filling the gaps in existing suite) or bug report or idea what could be added or done better.

All contributions should go through our GitHub repository. Bug reports, ideas or even questions should be raised by opening an issue on the GitHub tracker. Suggestions for changes in code or documentation should be submitted as a pull request. However, if you are not sure what to do, feel free to open an issue. All discussion will then take place on GitHub to keep the development of momepy transparent.

If you decide to contribute to the codebase, ensure that you are using an up-to-date `master` branch. The latest development version will always be there, including the documentation (powered by `sphinx`).

Details are available in the [documentation](https://docs.momepy.org/).

## Get in touch
If you have a question regarding momepy, feel free to open an issue on GitHub. Eventually, you can contact us on [dev@momepy.org](mailto:dev@momepy.org).

## Acknowledgements

Initial release of momepy was a result of a research of [Urban Design Studies Unit (UDSU)](http://udsu-strath.com) supported by the Axel and Margaret Ax:son Johnson Foundation as a part of “The Urban Form Resilience Project” in partnership with University of Strathclyde in Glasgow, UK. Further development was supported by [Geographic Data Science Lab](https://www.liverpool.ac.uk/geographic-data-science/) of the University of Liverpool wihtin [Urban Grammar AI](https://urbangrammarai.github.io) research project.

---
Copyright (c) 2018- Martin Fleischmann
