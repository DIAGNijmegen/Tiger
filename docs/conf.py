import sphinx_rtd_theme

import tiger

# Sphinx configuration
project = "Tiger"
copyright = "2019–2020, Diagnostic Image Analysis Group, Radboudumc"
author = "Nikolas Lessmann"
version = ".".join(tiger.__version__.split(".", 2)[:2])
release = tiger.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx-prompt",
]

master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Disable Google doc string
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Sort order for automatically documented class members
autodoc_member_order = "bysource"

# Extra dependencies might not be installed
autodoc_mock_imports = ["visdom", "torch"]

# Make the viewcode plugin follow imports
viewcode_follow_imported_members = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Show to do remarks?
todo_include_todos = True

# Theme for HTML documentation
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "prev_next_buttons_location": None,
    "sticky_navigation": True,
    "navigation_depth": 2,
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_use_index = False
html_favicon = "assets/favicon.ico"
