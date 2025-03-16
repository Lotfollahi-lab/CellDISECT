import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'CellDISECT'
copyright = '2024, Arian Amani, Stathis Megas'
author = 'Arian Amani, Stathis Megas'

# The full version, including alpha/beta/rc tags
release = '0.2.0b1'
version = '.'.join(release.split('.')[:2])  # Major.Minor version

# General configuration
language = 'en'
master_doc = 'index'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx_gallery.load_style',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'myst_parser',
    'sphinx.ext.githubpages',  # For GitHub Pages deployment
    'notfound.extension',  # For custom 404 pages
    'sphinx_sitemap',  # For generating sitemap
    'sphinx_multiversion',  # For versioning
    # New extensions
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'sphinx.ext.mathjax',
    'sphinx_design',
]

# Version configuration
smv_tag_whitelist = r'^v\d+\.\d+\.\d+.*$'  # Include version tags like v0.1.0, v0.2.0b1
smv_branch_whitelist = r'^(main|develop|beta-colab)$'  # Include main, develop, and beta branches
smv_remote_whitelist = r'^(origin)$'
smv_released_pattern = r'^tags/v\d+\.\d+\.\d+.*$'  # Released versions pattern
smv_outputdir_format = '{ref.name}'

# Myst Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Bibtex files
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# HTML output configuration
html_title = 'CellDISECT Documentation'
html_short_title = 'CellDISECT'
html_favicon = '_static/images/favicon.ico'
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Theme configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'body_max_width': 'none',  # Responsive layout
    'navigation_with_keys': True,  # Keyboard navigation
    'canonical_url': 'https://celldisect.readthedocs.io/en/latest/',
}

# Version display settings
html_context = {
    'display_version': True,
    'current_version': version,
    'versions': [
        ('latest', '/en/latest'),
        ('stable', '/en/stable'),
        (version, '')
    ]
}

# Search configuration
html_search_language = 'en'
html_search_options = {'type': 'default'}

# Add any paths that contain custom static files
html_static_path = ['_static']

# These paths are either relative to html_static_path or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

# Add changelog and contributing to the toctree
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html'
    ]
}

# 404 page configuration
notfound_context = {
    'title': 'Page Not Found',
    'body': '''
        <h1>Page Not Found</h1>
        <p>Sorry, we couldn't find that page.</p>
        <p>Try using the navigation menu or search box to find what you're looking for.</p>
    ''',
}
notfound_no_urls_prefix = True
notfound_template = '404.rst'

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_docstring_signature = True

# NBSphinx settings
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'
nbsphinx_prompt_width = 0


# Add notebooks to the patterns of files to copy
html_extra_path = ['robots.txt']

# Sitemap configuration
html_baseurl = 'https://celldisect.readthedocs.io/'
sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"

# Todo configuration
todo_include_todos = True
todo_emit_warnings = True
todo_link_only = False

# Graphviz configuration
graphviz_output_format = 'svg'

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    },
}

# Sphinx Design configuration
sd_fontawesome_latex = False

# Accessibility and image settings
suppress_warnings = ['image.nonlocal_uri']
images_config = {
    'default_image_width': '100%',
    'default_image_height': 'auto',
    'default_alt': 'Image in CellDISECT documentation',
}

# Better code block settings for accessibility
pygments_style = 'sphinx'
highlight_language = 'python3'

# Disable PDF and ePub builders
epub_build = False 