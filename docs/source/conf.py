# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from packaging.version import Version

sys.path.insert(0, os.path.abspath("../../"))

import momepy  # noqa

project = "momepy"
copyright = "2018-, Martin Fleischmann and PySAL Developers"  # noqa: A001
author = "Martin Fleischmann"

version = Version(momepy.__version__).public  # remove commit hash
release = version

language = "en"
html_title = project

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_immaterial",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"

templates_path = [
    "_templates",
]
exclude_patterns = []

intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "libpysal": (
        "https://pysal.org/libpysal/dev",
        None,
    ),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}
autoclass_content = "both"
html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/pysal_logo.svg"
html_favicon = "_static/pysal_favicon.ico"
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-code",
    },
    "site_url": "https://docs.momepy.org",
    "repo_url": "https://github.com/pysal/momepy/",
    "edit_uri": "blob/main/docs",
    "repo_name": "pysal/momepy",
    "features": [
        # "navigation.expand",
        # "navigation.tabs",
        # "navigation.tabs.sticky",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.footer",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        # "content.tabs.link",
        "content.code.copy",
        "content.action.edit",
        # "content.action.view",
        # "content.tooltips",
        # "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
    "version_dropdown": True,
    "version_json": "https://pysal.org/momepy/versions.json",
}
nb_execution_mode = "cache"
nb_execution_timeout = -1
nb_kernel_rgx_aliases = {".*": "python3"}
nb_merge_streams = True
autodoc_typehints = "none"


def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(momepy.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "momepy/%s#L%d-L%d" % find_source()  # noqa: UP031
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "main" if "dev" in release else ("v" + release)
    return f"https://github.com/pysal/momepy/blob/{tag}/{filename}"
