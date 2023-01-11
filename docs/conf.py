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
sys.path.insert(1, '_static/nodes')
# sys.path.insert(0, os.path.abspath('.'))

from docutils import nodes



# -- Project information -----------------------------------------------------

project = 'SYCLomatic'
copyright = 'Intel Corporation'
author = 'Intel'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx2dita']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_include_files']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/oneapi-src/SYCLomatic',
    'path_to_docs': 'docs',
    'use_issues_button': True,
    'use_edit_page_button': True,
    'repository_branch': 'SYCLomatic',
    'extra_footer': '<p align="right"><a href="https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html">Cookies</a></p>'


}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/oneAPI-rgb-rev-100.png'
html_favicon = '_static/favicons.png'
html_show_sourcelink = False
html_title = "SYCLomatic Documentation"

rst_epilog = """
.. include:: /_include_files/variables.txt
"""
