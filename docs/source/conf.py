# ──────────────────────────────────────────────────────────
#  Configuration file for the Sphinx documentation builder
# ──────────────────────────────────────────────────────────

import os
import sys
import datetime
from unittest.mock import MagicMock

# ── CRITICAL: Mock heavy dependencies FIRST ─────────────
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock all heavy dependencies before any other imports
MOCK_MODULES = [
    # Deep Learning / GPU
    "torch", "cupy", "numba", "numba.cuda", "numba.cuda.random", "numba.config",
    "pycuda", "pycuda.driver", "pycuda.autoinit", "tensorflow",
    
    # Robotics / Simulation
    "pybullet", "pybullet_data", "urchin", "urchin.urdf",
    
    # Computer Vision
    "cv2", "ultralytics", "opencv-python",
    
    # Scientific Computing (keep matplotlib for docs)
    "sklearn", "sklearn.cluster", "scipy", "scipy.spatial", "scipy.linalg",
    
    # Optional dependencies
    "plotly", "seaborn", "PIL", "Pillow",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# ── Path handling ────────────────────────────────────────
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)          # so `import ManipulaPy` works

# ── Project info ─────────────────────────────────────────
project   = "ManipulaPy"
author    = "Mohamed Aboelnar"
copyright = f"{datetime.datetime.now().year}, {author}"
version = release = "1.1.0"

# ── Core Extensions (always available) ──────────────────
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",        # .nojekyll file for GitHub Pages
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

# Try to add optional extensions - be conservative for RTD
_optional_extensions = {
    "myst_parser": try_add_extension("myst_parser"),
    "sphinx_copybutton": try_add_extension("sphinx_copybutton"),
    "sphinx_design": try_add_extension("sphinx_design"),
}

# Only add these if we're not on RTD (they may not be available)
if not os.environ.get('READTHEDOCS'):
    _optional_extensions.update({
        "sphinx.ext.autosectionlabel": try_add_extension("sphinx.ext.autosectionlabel"),
        "sphinx.ext.extlinks": try_add_extension("sphinx.ext.extlinks"), 
        "sphinx.ext.todo": try_add_extension("sphinx.ext.todo"),
        "sphinx.ext.ifconfig": try_add_extension("sphinx.ext.ifconfig"),
        "sphinx_tabs": try_add_extension("sphinx_tabs"), 
        "sphinx_togglebutton": try_add_extension("sphinx_togglebutton"),
        "sphinxext.opengraph": try_add_extension("sphinxext.opengraph"),
        "sphinx_sitemap": try_add_extension("sphinx_sitemap"),
    })

# ── Theme selection (RTD-compatible priority) ───────────
def _try_theme(name, mod_name, **kw):
    try:
        __import__(mod_name)
        return name, kw.get("extra", {})
    except ImportError:
        return None, {}

# Start with RTD theme for maximum compatibility
html_theme = "sphinx_rtd_theme" 
html_theme_options = {
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Try better themes if available (but fallback gracefully)
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Better HTML output options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True

# Only set these if files exist (prevent RTD errors)
logo_path = os.path.join(DOCS_DIR, "_static", "file.png")
if os.path.exists(logo_path):
    html_logo = "_static/file.png"
    html_favicon = "_static/file.png"

# ── LaTeX Configuration (RTD-compatible) ────────────────
# Use pdflatex for RTD compatibility (xelatex may not be available)
latex_engine = 'pdflatex'

# LaTeX document configuration - FIXED for RTD
latex_documents = [
    (
        'index',                    # master document (source file)
        'manipulapy.tex',          # output .tex filename  
        'ManipulaPy Documentation', # CLEAN title (no special chars)
        'Mohamed Aboelnar',        # author
        'manual',                  # document class
    ),
]

# Simplified LaTeX configuration for RTD compatibility
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage[utf8]{inputenc}
        \usepackage{newunicodechar}
        \DeclareUnicodeCharacter{1F680}{\textbf{[ROCKET]}}
        \DeclareUnicodeCharacter{2705}{\textbf{[CHECK]}}
        \DeclareUnicodeCharacter{274C}{\textbf{[X]}}
        \DeclareUnicodeCharacter{26A0}{\textbf{[WARNING]}}
        \DeclareUnicodeCharacter{1F4CB}{\textbf{[CLIPBOARD]}}
        \DeclareUnicodeCharacter{1F3A8}{\textbf{[ART]}}
        \DeclareUnicodeCharacter{1F4E6}{\textbf{[PACKAGE]}}
        \DeclareUnicodeCharacter{1F50C}{\textbf{[PLUG]}}
        \DeclareUnicodeCharacter{1F4CC}{\textbf{[PIN]}}
        \DeclareUnicodeCharacter{1F6A8}{\textbf{[ALERT]}}
        \DeclareUnicodeCharacter{1F50D}{\textbf{[SEARCH]}}
        \DeclareUnicodeCharacter{1F4A1}{\textbf{[IDEA]}}
        \DeclareUnicodeCharacter{1F527}{\textbf{[WRENCH]}}
        \DeclareUnicodeCharacter{2B50}{\textbf{[STAR]}}
        \DeclareUnicodeCharacter{1F3AF}{\textbf{[TARGET]}}
    ''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}

# ── Napoleon (NumPy / Google docstrings) ────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False  # Simplified for RTD
napoleon_type_aliases = {
    "array_like": ":term:`array_like`",
    "ndarray": "numpy.ndarray",
}

# ── Autodoc / Autosummary ───────────────────────────────
autodoc_default_options = {
    "members": True,
    "undoc_members": True,
    "special_members": "__init__",
    "show_inheritance": True,
    "member-order": "bysource",
    "exclude_members": "__weakref__",
}

# Better autodoc behavior
autodoc_typehints = "description"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"

autosummary_generate = True
autosummary_generate_overwrite = False  # Prevent RTD conflicts
autosummary_imported_members = False     # Simplify for RTD

# Mock imports - CRITICAL for RTD
autodoc_mock_imports = MOCK_MODULES

# ── Math support ─────────────────────────────────────────
# Use simpler mathjax config for RTD
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# ── Extension configuration ──────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ── Optional Extension Configurations ───────────────────
# External links shortcuts (only if extension is available)
if "sphinx.ext.extlinks" in extensions:
    extlinks = {
        "issue": ("https://github.com/boelnasr/ManipulaPy/issues/%s", "issue #%s"),
        "pull": ("https://github.com/boelnasr/ManipulaPy/pull/%s", "PR #%s"),
        "pypi": ("https://pypi.org/project/%s/", "%s"),
        "wiki": ("https://en.wikipedia.org/wiki/%s", "%s"),
    }

# TODO extension (only if available)
if "sphinx.ext.todo" in extensions:
    todo_include_todos = not os.environ.get('READTHEDOCS', False)  # Hide on RTD
    todo_emit_warnings = False  # Don't break builds

# Copy button configuration (only if extension is available)
if _optional_extensions.get("sphinx_copybutton", False):
    copybutton_prompt_text = r">>> |\.\.\. |\$ "
    copybutton_prompt_is_regexp = True

# OpenGraph / Social Media (only if extension is available)
if _optional_extensions.get("sphinxext.opengraph", False):
    ogp_site_url = "https://manipulapy.readthedocs.io/"
    ogp_site_name = "ManipulaPy Documentation"
    ogp_description_length = 200
    ogp_type = "website"

# Better section label generation (only if extension is available)
if "sphinx.ext.autosectionlabel" in extensions:
    autosectionlabel_prefix_document = True
    autosectionlabel_maxdepth = 3

# ── Quality of life improvements ────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Language and locale
language = "en"
today_fmt = "%B %d, %Y"

# Source file encoding
source_suffix = {".rst": None}

# Add markdown support if myst_parser is available
if _optional_extensions.get("myst_parser", False):
    source_suffix[".md"] = "myst_parser"

# Master document
master_doc = "index"

# ── Critical: Suppress warnings that break RTD builds ───
suppress_warnings = [
    'ref.citation',
    'ref.footnote', 
    'autosectionlabel.*',
    'toc.excluded',
    'epub.unknown_project_files',
    'app.add_directive',
]

# Don't fail on warnings
keep_warnings = True

def setup(app):
    """Sphinx setup hook for custom configuration."""
    
    # Always add our custom stylesheet
    app.add_css_file("custom.css")
    
    # Add custom JavaScript if it exists
    js_file = os.path.join(DOCS_DIR, "_static", "custom.js")
    if os.path.exists(js_file):
        app.add_js_file("custom.js")
    
    # Print configuration summary
    print("\n" + "="*50)
    print("📋 SPHINX BUILD CONFIGURATION")
    print("="*50)
    print(f"🎨 Theme: {html_theme}")
    print(f"📦 Core extensions: {len(extensions)}")
    print(f"🔌 Optional extensions: {sum(_optional_extensions.values())}")
    available = [k for k, v in _optional_extensions.items() if v]
    if available:
        print(f"   Available: {', '.join(available)}")
    print(f"📄 LaTeX engine: {latex_engine}")
    print(f"🏗️  RTD Build: {bool(os.environ.get('READTHEDOCS'))}")
    print("="*50)
    
    # Attempt to import ManipulaPy with comprehensive error handling
    try:
        import ManipulaPy
        print("✅ ManipulaPy imported successfully")
        print(f"   Version: {getattr(ManipulaPy, '__version__', 'unknown')}")
        print(f"   Location: {ManipulaPy.__file__}")
        
        # Check key modules
        key_modules = ["kinematics", "dynamics", "control", "path_planning"]
        for module_name in key_modules:
            try:
                module = getattr(ManipulaPy, module_name, None)
                if module:
                    print(f"   ✅ {module_name} available")
                else:
                    print(f"   ⚠️  {module_name} not found")
            except Exception as e:
                print(f"   ❌ Error checking {module_name}: {e}")
                
    except ImportError as exc:
        print(f"⚠️  ManipulaPy import failed: {exc}")
        print("    This is expected during RTD builds with heavy dependencies")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("="*50 + "\n")