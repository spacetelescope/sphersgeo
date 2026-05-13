# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import tomllib
import datetime
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as metadata_file:
    metadata = tomllib.load(metadata_file)["project"]

project = metadata["name"]
author = "Space Telescope Science Institute"
copyright = f"{datetime.datetime.today().year}, {author}"

package = importlib.import_module(metadata["name"])
try:
    version = package.__version__.split("-", 1)[0]
    # The full version, including alpha/beta/rc tags.
    release = package.__version__
except AttributeError:
    version = "dev"
    release = "dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_inline_tabs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# reST default role used for single backticks (`text`)
default_role = "obj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "stsci_pri_combo_mark_light_bkgd.png",
    "dark_logo": "stsci_pri_combo_mark_dark_bkgd.png",
}

# # -- sphinx-autoapi configuration --------------------------------------------
# # https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = ["../src/python/sphersgeo/"]
autoapi_root = "api"
autoapi_generate_api_docs = False
autoapi_member_order = "bysource"
autoapi_python_class_content = "both"

# -- sphinx.ext.intersphinx configuration --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "asdf": ("https://www.asdf-format.org/projects/asdf/en/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "gwcs": ("https://gwcs.readthedocs.io/en/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
}

# -- sphinx.ext.autodoc configuration --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
