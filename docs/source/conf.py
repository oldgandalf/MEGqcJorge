# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MEG QC'
copyright = '2023, ANCP Lab, University of Oldenburg, Evgeniia Gapontseva, Aaron Reer, Jochem Rieger'
author = 'Evgeniia Gapontseva, Aaron Reer, Jochem Rieger'
release = '2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



extensions = ['sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_inline_tabs'
    ]

# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'

# alabaster theme options
html_theme_options = {
    "fixed_sidebar": "false",
    "github_button": "true",
    "sidebarwidth": "400",
    "body_min_width": "800",
    "github_user": "ANCPLabOldenburg",
    "github_repo": "MEG-QC-code",
    "description": "Python based pipeline for quality control of MEG data",
    # "logo": "meg_qc_logo.png",
    # "logo_name": "true",
}

html_static_path = ['_static']
