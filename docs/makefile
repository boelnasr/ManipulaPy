# ManipulaPy Documentation Makefile
# Robust Sphinx documentation build system
# Copyright (c) 2025 Mohamed Aboelnar

# =============================================================================
# Configuration
# =============================================================================

# Sphinx configuration
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGE_DIR   = ../ManipulaPy

# Python configuration
PYTHON        ?= python3
PIP           ?= pip3

# Headless / CI-friendly env for all sphinx calls
SPHINXENV     = MPLBACKEND=Agg SPHINX_BUILD=1

# Build strictness
#   0 = default (keep going, do not fail on warnings)
#   1 = treat warnings as errors (-W)
SPHINXSTRICT  ?= 0

# Baseline Sphinx options: parallel (-j auto), show full tracebacks (-T)
SPHINXOPTS_COMMON = -j auto -T

# Strict toggle
ifeq ($(SPHINXSTRICT),1)
	SPHINXOPTS := $(SPHINXOPTS_COMMON) -W
else
	SPHINXOPTS := $(SPHINXOPTS_COMMON) --keep-going
endif

# Optional fixer script that inserts blank lines after headers (safe if missing)
DOCS_FIXER ?= fix_docs.py

# Colors for output
RED    = \033[0;31m
GREEN  = \033[0;32m
YELLOW = \033[0;33m
BLUE   = \033[0;34m
NC     = \033[0m # No Color

# =============================================================================
# Help target
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)ManipulaPy Documentation Build System$(NC)"
	@echo "$(BLUE)====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)html$(NC)            Build HTML documentation (non-strict)"
	@echo "  $(YELLOW)html-strict$(NC)     Build HTML with -W (warnings = errors)"
	@echo "  $(YELLOW)dirhtml$(NC)         Build HTML pages as directories"
	@echo "  $(YELLOW)singlehtml$(NC)      Build single HTML page"
	@echo "  $(YELLOW)pickle$(NC)          Build pickle documentation"
	@echo "  $(YELLOW)json$(NC)            Build JSON documentation"
	@echo "  $(YELLOW)htmlhelp$(NC)        Build HTML help documentation"
	@echo "  $(YELLOW)qthelp$(NC)          Build Qt help documentation"
	@echo "  $(YELLOW)devhelp$(NC)         Build DevHelp documentation"
	@echo "  $(YELLOW)epub$(NC)            Build EPUB documentation"
	@echo "  $(YELLOW)latex$(NC)           Build LaTeX documentation"
	@echo "  $(YELLOW)latexpdf$(NC)        Build PDF documentation via LaTeX"
	@echo "  $(YELLOW)latexpdfja$(NC)      Build PDF documentation via LaTeX (Japanese)"
	@echo "  $(YELLOW)text$(NC)            Build plain text documentation"
	@echo "  $(YELLOW)man$(NC)             Build manual pages"
	@echo "  $(YELLOW)texinfo$(NC)         Build Texinfo documentation"
	@echo "  $(YELLOW)info$(NC)            Build info documentation"
	@echo "  $(YELLOW)gettext$(NC)         Build gettext documentation"
	@echo "  $(YELLOW)changes$(NC)         Build changes documentation"
	@echo "  $(YELLOW)xml$(NC)             Build XML documentation"
	@echo "  $(YELLOW)pseudoxml$(NC)       Build pseudo-XML documentation"
	@echo ""
	@echo "$(GREEN)Custom targets:$(NC)"
	@echo "  $(YELLOW)clean$(NC)           Clean build directory"
	@echo "  $(YELLOW)clean-all$(NC)       Clean build directory and API docs"
	@echo "  $(YELLOW)api$(NC)             Generate API documentation"
	@echo "  $(YELLOW)fmt$(NC)             Normalize list spacing in .rst/.md"
	@echo "  $(YELLOW)serve$(NC)           Serve documentation locally"
	@echo "  $(YELLOW)install-deps$(NC)    Install documentation dependencies"
	@echo "  $(YELLOW)check-deps$(NC)      Check if dependencies are installed"
	@echo "  $(YELLOW)rebuild$(NC)         Clean and rebuild HTML documentation"
	@echo "  $(YELLOW)rtd$(NC)             Build as Read the Docs (READTHEDOCS=1)"
	@echo "  $(YELLOW)nitpick$(NC)         Nitpicky run (-n) to catch bad refs"
	@echo "  $(YELLOW)warnings.log$(NC)    Write build warnings to a log file"

# =============================================================================
# Dependency management
# =============================================================================

.PHONY: check-sphinx
check-sphinx:
	@command -v $(SPHINXBUILD) >/dev/null 2>&1 || { \
		echo "$(RED)Error: sphinx-build not found. Please install Sphinx.$(NC)"; \
		echo "$(YELLOW)Run: $(PIP) install sphinx$(NC)"; \
		exit 1; \
	}

.PHONY: check-deps
check-deps: check-sphinx ## Check if dependencies are installed
	@echo "$(BLUE)Checking documentation dependencies...$(NC)"
	@$(PYTHON) -c "import sphinx; print('Sphinx version:', sphinx.__version__)"
	@$(PYTHON) -c "import sphinx_rtd_theme; print('RTD theme: OK')" 2>/dev/null || \
		echo "$(YELLOW)Warning: sphinx_rtd_theme not found$(NC)"
	@$(PYTHON) -c "import myst_parser; print('MyST parser: OK')" 2>/dev/null || \
		echo "$(YELLOW)Warning: myst_parser not found$(NC)"
	@echo "$(GREEN)Dependency check completed$(NC)"

.PHONY: install-deps
install-deps: ## Install documentation dependencies
	@echo "$(BLUE)Installing documentation dependencies...$(NC)"
	$(PIP) install sphinx sphinx_rtd_theme myst-parser sphinx-autoapi sphinx-copybutton sphinx-design sphinx-autobuild
	@echo "$(GREEN)Documentation dependencies installed$(NC)"

# =============================================================================
# Formatting helpers
# =============================================================================

.PHONY: fmt
fmt: ## Normalize header/list spacing (optional fixer script)
	@echo "$(BLUE)Running docs fixer (if present)...$(NC)"
	@if [ -f "$(DOCS_FIXER)" ]; then \
		$(PYTHON) $(DOCS_FIXER) --write; \
	elif [ -f "scripts/$(DOCS_FIXER)" ]; then \
		$(PYTHON) scripts/$(DOCS_FIXER) --write; \
	else \
		echo "$(YELLOW)No fixer script found (skipping)$(NC)"; \
	fi

# =============================================================================
# API Documentation
# =============================================================================

.PHONY: api
api: check-sphinx ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@if [ -d "$(PACKAGE_DIR)" ]; then \
		$(SPHINXAPIDOC) -o $(SOURCEDIR)/api $(PACKAGE_DIR) \
			--force --module-first --no-toc -e; \
		echo "$(GREEN)API documentation generated$(NC)"; \
	else \
		echo "$(RED)Error: Package directory $(PACKAGE_DIR) not found$(NC)"; \
		exit 1; \
	fi

# =============================================================================
# Build targets
# =============================================================================

.PHONY: clean
clean: ## Clean build directory
	@echo "$(BLUE)Cleaning build directory...$(NC)"
	rm -rf $(BUILDDIR)/*
	@echo "$(GREEN)Build directory cleaned$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean build directory and API docs
	@echo "$(BLUE)Cleaning API documentation...$(NC)"
	rm -rf $(SOURCEDIR)/api
	@echo "$(GREEN)API documentation cleaned$(NC)"

.PHONY: html
html: check-sphinx ## Build HTML documentation (non-strict by default)
	@echo "$(BLUE)Building HTML documentation...$(NC)"
	@mkdir -p $(BUILDDIR)
	@$(SPHINXENV) $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "$(GREEN)HTML documentation built in $(BUILDDIR)/html$(NC)"
	@echo "$(YELLOW)Open $(BUILDDIR)/html/index.html in your browser$(NC)"

.PHONY: html-strict
html-strict: ## Build HTML with warnings as errors
	@$(MAKE) html SPHINXSTRICT=1

.PHONY: rebuild
rebuild: clean-all fmt api html ## Clean and rebuild HTML documentation
	@echo "$(GREEN)Documentation rebuilt successfully$(NC)"

.PHONY: serve
serve: html ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop the server$(NC)"
	cd $(BUILDDIR)/html && $(PYTHON) -m http.server 8000

# =============================================================================
# Other Sphinx targets
# =============================================================================

.PHONY: dirhtml
dirhtml: check-sphinx ## Build HTML pages as directories
	@echo "$(BLUE)Building dirhtml documentation...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml $(SPHINXOPTS)
	@echo "$(GREEN)dirhtml documentation built in $(BUILDDIR)/dirhtml$(NC)"

.PHONY: singlehtml
singlehtml: check-sphinx ## Build single HTML page
	@echo "$(BLUE)Building single HTML documentation...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b singlehtml $(SOURCEDIR) $(BUILDDIR)/singlehtml $(SPHINXOPTS)
	@echo "$(GREEN)Single HTML documentation built in $(BUILDDIR)/singlehtml$(NC)"

.PHONY: pickle
pickle: check-sphinx ## Build pickle documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b pickle $(SOURCEDIR) $(BUILDDIR)/pickle $(SPHINXOPTS)

.PHONY: json
json: check-sphinx ## Build JSON documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b json $(SOURCEDIR) $(BUILDDIR)/json $(SPHINXOPTS)

.PHONY: htmlhelp
htmlhelp: check-sphinx ## Build HTML help documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b htmlhelp $(SOURCEDIR) $(BUILDDIR)/htmlhelp $(SPHINXOPTS)

.PHONY: qthelp
qthelp: check-sphinx ## Build Qt help documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b qthelp $(SOURCEDIR) $(BUILDDIR)/qthelp $(SPHINXOPTS)

.PHONY: devhelp
devhelp: check-sphinx ## Build DevHelp documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b devhelp $(SOURCEDIR) $(BUILDDIR)/devhelp $(SPHINXOPTS)

.PHONY: epub
epub: check-sphinx ## Build EPUB documentation
	@echo "$(BLUE)Building EPUB documentation...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub $(SPHINXOPTS)
	@echo "$(GREEN)EPUB documentation built in $(BUILDDIR)/epub$(NC)"

.PHONY: latex
latex: check-sphinx ## Build LaTeX documentation
	@echo "$(BLUE)Building LaTeX documentation...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	@echo "$(GREEN)LaTeX documentation built in $(BUILDDIR)/latex$(NC)"

.PHONY: latexpdf
latexpdf: latex ## Build PDF documentation via LaTeX
	@echo "$(BLUE)Building PDF documentation...$(NC)"
	@if command -v make >/dev/null 2>&1; then \
		cd $(BUILDDIR)/latex && make all-pdf; \
		echo "$(GREEN)PDF documentation built in $(BUILDDIR)/latex$(NC)"; \
	else \
		echo "$(RED)Error: make not found. Cannot build PDF.$(NC)"; \
		exit 1; \
	fi

.PHONY: latexpdfja
latexpdfja: check-sphinx ## Build PDF documentation via LaTeX (Japanese)
	@$(SPHINXENV) $(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	cd $(BUILDDIR)/latex && make all-pdf-ja

.PHONY: text
text: check-sphinx ## Build plain text documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b text $(SOURCEDIR) $(BUILDDIR)/text $(SPHINXOPTS)

.PHONY: man
man: check-sphinx ## Build manual pages
	@$(SPHINXENV) $(SPHINXBUILD) -b man $(SOURCEDIR) $(BUILDDIR)/man $(SPHINXOPTS)

.PHONY: texinfo
texinfo: check-sphinx ## Build Texinfo documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b texinfo $(SOURCEDIR) $(BUILDDIR)/texinfo $(SPHINXOPTS)

.PHONY: info
info: texinfo ## Build info documentation
	cd $(BUILDDIR)/texinfo && make info

.PHONY: gettext
gettext: check-sphinx ## Build gettext documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/locale $(SPHINXOPTS)

.PHONY: changes
changes: check-sphinx ## Build changes documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b changes $(SOURCEDIR) $(BUILDDIR)/changes $(SPHINXOPTS)

.PHONY: xml
xml: check-sphinx ## Build XML documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b xml $(SOURCEDIR) $(BUILDDIR)/xml $(SPHINXOPTS)

.PHONY: pseudoxml
pseudoxml: check-sphinx ## Build pseudo-XML documentation
	@$(SPHINXENV) $(SPHINXBUILD) -b pseudoxml $(SOURCEDIR) $(BUILDDIR)/pseudoxml $(SPHINXOPTS)

# =============================================================================
# Quality assurance targets
# =============================================================================

.PHONY: linkcheck
linkcheck: check-sphinx ## Check all external links
	@echo "$(BLUE)Checking external links...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)
	@echo "$(GREEN)Link check completed. See $(BUILDDIR)/linkcheck/output.txt$(NC)"

.PHONY: doctest
doctest: check-sphinx ## Run doctests in documentation
	@echo "$(BLUE)Running doctests...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS)
	@echo "$(GREEN)Doctests completed$(NC)"

.PHONY: coverage
coverage: check-sphinx ## Check documentation coverage
	@echo "$(BLUE)Checking documentation coverage...$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage $(SPHINXOPTS)
	@echo "$(GREEN)Coverage check completed. See $(BUILDDIR)/coverage/$(NC)"

.PHONY: nitpick
nitpick: ## Run sphinx with nitpicky mode (-n) to catch bad refs
	@$(MAKE) html SPHINXOPTS="$(SPHINXOPTS) -n"

.PHONY: warnings.log
warnings.log: ## Build and write warnings to a log file (non-strict)
	@mkdir -p $(BUILDDIR)
	@echo "$(BLUE)Building and capturing warnings to $(BUILDDIR)/warnings.log$(NC)"
	@$(SPHINXENV) $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) 2> $(BUILDDIR)/warnings.log || true
	@echo "$(YELLOW)See $(BUILDDIR)/warnings.log$(NC)"

# =============================================================================
# Live-reload / watch
# =============================================================================

.PHONY: watch
watch: html ## Watch for changes and rebuild (requires entr)
	@echo "$(BLUE)Watching for changes...$(NC)"
	@command -v entr >/dev/null 2>&1 || { \
		echo "$(RED)Error: entr not found. Install with: apt install entr$(NC)"; \
		exit 1; \
	}
	find $(SOURCEDIR) -name "*.rst" -o -name "*.md" -o -name "*.py" | entr $(MAKE) html

.PHONY: live
live: ## Live reload server (requires sphinx-autobuild)
	@echo "$(BLUE)Starting live reload server...$(NC)"
	@$(PYTHON) -c "import sphinx_autobuild" 2>/dev/null || { \
		echo "$(RED)Error: sphinx-autobuild not found.$(NC)"; \
		echo "$(YELLOW)Install with: $(PIP) install sphinx-autobuild$(NC)"; \
		exit 1; \
	}
	$(SPHINXENV) sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) --host 0.0.0.0 --port 8000

# =============================================================================
# RTD-like build
# =============================================================================

.PHONY: rtd
rtd: ## Build with READTHEDOCS=1 to mimic RTD environment
	@echo "$(BLUE)Building as if on Read the Docs (READTHEDOCS=1)...$(NC)"
	@READTHEDOCS=1 $(SPHINXENV) $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

# =============================================================================
# Default and special targets
# =============================================================================

# Default target
.DEFAULT_GOAL := help

# Catch-all target for any unmatched targets (sphinx -M dispatcher)
%: Makefile check-sphinx
	@$(SPHINXENV) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
