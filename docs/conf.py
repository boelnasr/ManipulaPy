# ──────────────────────────────────────────────────────────
#  Configuration file for the Sphinx documentation builder
# ──────────────────────────────────────────────────────────

import os
import sys
import datetime

# ── Path handling ────────────────────────────────────────
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)          # so `import ManipulaPy` works

# ── Project info ─────────────────────────────────────────
project   = "ManipulaPy"
author    = "Mohamed Aboelnar"
copyright = f"{datetime.datetime.now().year}, {author}"
version = release = "1.2.0"

# ── Core Extensions (always available) ──────────────────
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    
    # Enhanced documentation features
    "sphinx.ext.autosectionlabel",   # every heading gets an explicit label
    "sphinx.ext.githubpages",        # .nojekyll file for GitHub Pages
    "sphinx.ext.extlinks",           # external link shortcuts
    "sphinx.ext.todo",               # TODO directives
    "sphinx.ext.ifconfig",           # conditional content
    
    # Graphviz for block diagrams
    "sphinx.ext.graphviz",           # dot/graphviz diagram support
    
    # Third-party extensions (check availability)
    "myst_parser",                   # markdown support
    "sphinx_design",                 # nice HTML components (<sd-*)
]

# ── Safely add optional extensions ──────────────────────
def try_add_extension(ext_name):
    """Safely try to add an extension."""
    try:
        __import__(ext_name)
        extensions.append(ext_name)
        print(f"✅ Added optional extension: {ext_name}")
        return True
    except ImportError:
        print(f"⚠️  Optional extension not available: {ext_name}")
        return False

# Try to add optional extensions
_optional_extensions = {
    "sphinx_copybutton": try_add_extension("sphinx_copybutton"),
    "sphinx_tabs": try_add_extension("sphinx_tabs"), 
    "sphinx_togglebutton": try_add_extension("sphinx_togglebutton"),
    "sphinxext.opengraph": try_add_extension("sphinxext.opengraph"),
    "sphinx_sitemap": try_add_extension("sphinx_sitemap"),
}

# ── Theme selection (furo -> pydata -> RTD -> default) ──
def _try_theme(name, mod_name, **kw):
    try:
        mod = __import__(mod_name)
        return name, kw.get("extra", {})
    except ImportError:
        return None, {}

html_theme, html_theme_options = "default", {}
for _name, _mod, _opts in [
    ("furo", "furo", {"extra": {      # first choice - modern and clean
        "navigation_with_keys": True,
        "top_of_page_button": "edit",
        "source_repository": "https://github.com/boelnasr/ManipulaPy/",
        "source_branch": "main",
        "source_directory": "docs/source/",
        "light_css_variables": {
            "color-brand-primary": "#2980B9",
            "color-brand-content": "#2980B9",
        },
        "dark_css_variables": {
            "color-brand-primary": "#4FC3F7",
            "color-brand-content": "#4FC3F7",
        },
    }}),
    ("pydata_sphinx_theme", "pydata_sphinx_theme", {
        "extra": {
            "github_url": "https://github.com/boelnasr/ManipulaPy",
            "navbar_start": ["navbar-logo", "version-dropdown"],
            "navbar_end": ["search-field.html", "navbar-icon-links"],
            "navbar_persistent": ["search-button"],
            "footer_items": ["copyright", "sphinx-version"],
            "collapse_navigation": True,
            "show_prev_next": False,
            "icon_links": [
                {
                    "name": "GitHub",
                    "url": "https://github.com/boelnasr/ManipulaPy",
                    "icon": "fab fa-github-square",
                    "type": "fontawesome",
                },
                {
                    "name": "PyPI",
                    "url": "https://pypi.org/project/manipulapy/",
                    "icon": "fas fa-box",
                    "type": "fontawesome",
                },
            ],
        }
    }),
    ("sphinx_rtd_theme", "sphinx_rtd_theme", {
        "extra": {
            "style_nav_header_background": "#2980B9",
            "collapse_navigation": True,
            "navigation_depth": 4,
            "includehidden": True,
            "titles_only": False,
        }
    }),
]:
    _selected, _opts_dict = _try_theme(_name, _mod, extra=_opts["extra"])
    if _selected:
        html_theme = _selected
        html_theme_options.update(_opts_dict)
        print(f"✅ Using theme: {_selected}")
        break

# ── HTML output ──────────────────────────────────────────
html_title = f"{project} {release} Documentation"
html_short_title = f"{project} Docs"
html_logo  = "_static/file.png"  # Consider adding a proper logo
html_favicon = "_static/file.png"  # Add a favicon for better branding
html_static_path = ["_static"]
html_css_files  = ["custom.css"]

# Better HTML output options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True
html_use_opensearch = f"{project} Documentation"

# ── LaTeX Configuration (FIXED) ──────────────────────────
# Use XeLaTeX for better Unicode support
latex_engine = 'xelatex'

# LaTeX document configuration - PROPERLY FIXED
latex_documents = [
    (
        'index',                    # master document (source file)
        'manipulapy.tex',          # output .tex filename  
        'ManipulaPy User Guide',   # CLEAN title (no underscores or special chars)
        'Mohamed Aboelnar',        # author
        'manual',                  # document class
    ),
]

# LaTeX configuration for Unicode and better formatting
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
        % Basic Unicode support with fontspec
        \usepackage{fontspec}
        \setmainfont{DejaVu Serif}
        \setsansfont{DejaVu Sans}
        \setmonofont{DejaVu Sans Mono}
        
        % Handle specific Unicode characters that commonly cause issues
        \usepackage{newunicodechar}
        \newunicodechar{🚀}{\textbf{[ROCKET]}}
        \newunicodechar{✅}{\textbf{[CHECK]}}
        \newunicodechar{❌}{\textbf{[X]}}
        \newunicodechar{⚠}{\textbf{[WARNING]}}
        \newunicodechar{📋}{\textbf{[CLIPBOARD]}}
        \newunicodechar{🎨}{\textbf{[ART]}}
        \newunicodechar{📦}{\textbf{[PACKAGE]}}
        \newunicodechar{🔌}{\textbf{[PLUG]}}
        \newunicodechar{📌}{\textbf{[PIN]}}
        \newunicodechar{🚨}{\textbf{[ALERT]}}
        \newunicodechar{🔍}{\textbf{[SEARCH]}}
        \newunicodechar{💡}{\textbf{[IDEA]}}
        \newunicodechar{🔧}{\textbf{[WRENCH]}}
        \newunicodechar{⭐}{\textbf{[STAR]}}
        \newunicodechar{🎯}{\textbf{[TARGET]}}
    ''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
    'sphinxsetup': '''
        hmargin={1in,1in},
        vmargin={1in,1in},
        verbatimwithframe=true,
        TitleColor={rgb}{0,0,0},
        HeaderFamily=\\rmfamily\\bfseries,
        InnerLinkColor={rgb}{0.2,0.4,0.8},
        OuterLinkColor={rgb}{0.8,0.2,0.2}
    ''',
}

# LaTeX additional settings
latex_use_xindy = False
latex_domain_indices = True
latex_show_pagerefs = True
latex_show_urls = 'footnote'

# ── Graphviz Configuration ───────────────────────────────
# Graphviz (dot) configuration for block diagrams
graphviz_dot = 'dot'  # Path to dot executable (usually in PATH)
graphviz_dot_args = ['-Grankdir=TB', '-Gsize="8,10"', '-Gdpi=150']
graphviz_output_format = 'svg'  # Use SVG for better quality

# Default attributes for all graphviz diagrams
graphviz_default_attrs = {
    'graph': {
        'rankdir': 'TB',
        'bgcolor': 'transparent',
        'fontname': 'Arial',
        'fontsize': '10',
        'nodesep': '0.5',
        'ranksep': '0.8',
    },
    'node': {
        'shape': 'box',
        'style': 'rounded,filled',
        'fillcolor': 'lightblue',
        'fontname': 'Arial',
        'fontsize': '9',
        'color': 'black',
        'penwidth': '1',
    },
    'edge': {
        'fontname': 'Arial',
        'fontsize': '8',
        'color': 'black',
        'penwidth': '1',
        'arrowsize': '0.8',
    },
}

# ── Napoleon (NumPy / Google docstrings) ────────────────
napoleon_google_docstring      = True
napoleon_numpy_docstring       = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array_like": ":term:`array_like`",
    "ndarray": "numpy.ndarray",
    "DataFrame": "pandas.DataFrame",
}

# ── Autodoc / Autosummary ───────────────────────────────
autodoc_default_options = dict(
    members=True,
    undoc_members=True,
    special_members="__init__",
    show_inheritance=True,
    member_order="bysource",
    exclude_members="__weakref__",
    inherited_members=True,
)

# Better autodoc behavior
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

# Mock heavy / optional deps
autodoc_mock_imports = [
    # Deep Learning / GPU
    "torch", "cupy", "numba", "numba.cuda", "pycuda", "pycuda.driver",
    "pycuda.autoinit", "tensorflow",
    
    # Robotics / Simulation
    "pybullet", "pybullet_data", "urchin", "urchin.urdf",
    
    # Computer Vision
    "cv2", "ultralytics", "opencv-python",
    
    # Scientific Computing
    "sklearn", "sklearn.cluster", "scipy", "scipy.spatial",
    "matplotlib", "matplotlib.pyplot",
    
    # Optional dependencies
    "plotly", "seaborn", "PIL", "Pillow",
]

# ── Math support ─────────────────────────────────────────
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}

# ── Extension configuration ──────────────────────────────
intersphinx_mapping = {
    "python":      ("https://docs.python.org/3/", None),
    "numpy":       ("https://numpy.org/doc/stable/", None),
    "scipy":       ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib":  ("https://matplotlib.org/stable/", None),
    "sklearn":     ("https://scikit-learn.org/stable/", None),
    "torch":       ("https://pytorch.org/docs/stable/", None),
    "cv2":         ("https://docs.opencv.org/4.x/", None),
}

# External links shortcuts
extlinks = {
    "issue": ("https://github.com/boelnasr/ManipulaPy/issues/%s", "issue #%s"),
    "pull": ("https://github.com/boelnasr/ManipulaPy/pull/%s", "PR #%s"),
    "pypi": ("https://pypi.org/project/%s/", "%s"),
    "wiki": ("https://en.wikipedia.org/wiki/%s", "%s"),
}

# TODO extension
todo_include_todos = True
todo_emit_warnings = True

# ── Optional Extension Configurations ───────────────────
# Copy button configuration (only if extension is available)
if _optional_extensions.get("sphinx_copybutton", False):
    copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
    copybutton_prompt_is_regexp = True
    copybutton_line_continuation_character = "\\"
    copybutton_here_doc_delimiter = "EOT"

# OpenGraph / Social Media (only if extension is available)
if _optional_extensions.get("sphinxext.opengraph", False):
    ogp_site_url = "https://boelnasr.github.io/ManipulaPy/"
    ogp_site_name = "ManipulaPy Documentation"
    ogp_description_length = 200
    ogp_type = "website"
    ogp_image = "_static/file.png"

# Sitemap generation (only if extension is available)
if _optional_extensions.get("sphinx_sitemap", False):
    html_baseurl = "https://boelnasr.github.io/ManipulaPy/"
    sitemap_url_scheme = "{link}"

# ── Source code links ───────────────────────────────────
viewcode_enable_epub = True

# ── Quality of life improvements ────────────────────────
# Better section label generation
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

# Show todos in development builds
import os
if os.environ.get("SPHINX_BUILD_TYPE") == "dev":
    todo_include_todos = True
else:
    todo_include_todos = False

# ── Misc / setup hook ───────────────────────────────────
templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Language and locale
language = "en"
today_fmt = "%B %d, %Y"

# Source file encoding
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

def setup(app):
    """Sphinx setup hook for custom configuration."""
    
    # Always add our custom stylesheet
    app.add_css_file("custom.css")
    
    # Add custom JavaScript if it exists
    js_file = os.path.join(DOCS_DIR, "_static", "custom.js")
    if os.path.exists(js_file):
        app.add_js_file("custom.js")
    
    # Add version info to environment
    app.add_config_value("project_version", version, "env")
    
    # Check if Graphviz is available
    import subprocess
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        print("✅ Graphviz (dot) is available for block diagrams")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Graphviz (dot) not found. Install with: sudo apt-get install graphviz")
        print("    Block diagrams will not render properly without Graphviz")
    
    # Friendly import check with more detailed diagnostics
    try:
        import ManipulaPy
        print("✅ ManipulaPy imported successfully during docs build")
        print(f"   Version: {getattr(ManipulaPy, '__version__', 'unknown')}")
        print(f"   Location: {ManipulaPy.__file__}")
        
        # Check for key modules
        key_modules = ["kinematics", "dynamics", "control", "path_planning"]
        for module_name in key_modules:
            try:
                module = getattr(ManipulaPy, module_name, None)
                if module:
                    print(f"   ✅ {module_name} module available")
                else:
                    print(f"   ⚠️  {module_name} module not found")
            except Exception as e:
                print(f"   ❌ Error checking {module_name}: {e}")
                
    except ImportError as exc:
        print(f"⚠️  ManipulaPy not importable during docs build: {exc}")
        print("    This is normal if you're building docs without installing the package")
        print("    Run `pip install -e .` from the repo root if needed")
    except Exception as e:
        print(f"❌ Unexpected error importing ManipulaPy: {e}")

# ── Development helpers ──────────────────────────────────
def print_build_summary():
    """Print a summary of the build configuration."""
    print("\n" + "="*50)
    print("📋 SPHINX BUILD CONFIGURATION SUMMARY")
    print("="*50)
    print(f"🎨 Theme: {html_theme}")
    print(f"📦 Core extensions: {len([e for e in extensions if not e.startswith('sphinx_copybutton')])}")
    print(f"🔌 Optional extensions: {sum(_optional_extensions.values())}/{len(_optional_extensions)}")
    available = [k for k, v in _optional_extensions.items() if v]
    if available:
        print(f"   Available: {', '.join(available)}")
    print(f"📊 Graphviz enabled: {'sphinx.ext.graphviz' in extensions}")
    print(f"📄 LaTeX engine: {latex_engine}")
    print("="*50 + "\n")

# Call summary (uncomment for debugging)
print_build_summary()
