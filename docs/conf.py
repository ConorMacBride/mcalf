# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'MCALF'
author = 'Conor D. MacBride & David B. Jess'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

# The full version, including alpha/beta/rc tags
from pkg_resources import get_distribution
release = get_distribution('mcalf').version


# -- General configuration ---------------------------------------------------

import sphinx_rtd_theme
from sphinx_gallery.sorting import FileNameSortKey

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]
numpydoc_show_class_members = False
nbsphinx_execute = 'never'
automodsumm_private_methods_of = ['mcalf.models.ModelBase']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    "python": ('https://docs.python.org/3', None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ('https://matplotlib.org/', None),
    "numpy": ('https://numpy.org/doc/stable', None),
    "scipy": ('https://docs.scipy.org/doc/scipy/reference', None),
    "sklearn": ('https://scikit-learn.org/stable', None),
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': ['../examples/gallery'],
    # path to where to save gallery generated output
    'gallery_dirs': ['gallery'],
    # specify that examples should be ordered according to filename
    'within_subsection_order': FileNameSortKey,
    # directory where function/class granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created.
    'doc_module': ('mcalf',),
    # don't print e.g. sphinx_gallery_thumbnail_number = 2
    'remove_config_comments': True,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
