# ManipulaPy Documentation Makefile
# Robust Sphinx documentation build system
# Copyright (c) 2025 Mohamed Aboelnar

# =============================================================================
# Configuration
# =============================================================================

# Sphinx configuration
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGE_DIR   = ../ManipulaPy

# Python configuration
PYTHON        ?= python3
PIP           ?= pip3

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
	@echo "  $(YELLOW)html$(NC)          Build HTML documentation"
	@echo "  $(YELLOW)dirhtml$(NC)       Build HTML pages as directories"
	@echo "  $(YELLOW)singlehtml$(NC)    Build single HTML page"
	@echo "  $(YELLOW)pickle$(NC)        Build pickle documentation"
	@echo "  $(YELLOW)json$(NC)          Build JSON documentation"
	@echo "  $(YELLOW)htmlhelp$(NC)      Build HTML help documentation"
	@echo "  $(YELLOW)qthelp$(NC)        Build Qt help documentation"
	@echo "  $(YELLOW)devhelp$(NC)       Build DevHelp documentation"
	@echo "  $(YELLOW)epub$(NC)          Build EPUB documentation"
	@echo "  $(YELLOW)latex$(NC)         Build LaTeX documentation"
	@echo "  $(YELLOW)latexpdf$(NC)      Build PDF documentation via LaTeX"
	@echo "  $(YELLOW)latexpdfja$(NC)    Build PDF documentation via LaTeX (Japanese)"
	@echo "  $(YELLOW)text$(NC)          Build plain text documentation"
	@echo "  $(YELLOW)man$(NC)           Build manual pages"
	@echo "  $(YELLOW)texinfo$(NC)       Build Texinfo documentation"
	@echo "  $(YELLOW)info$(NC)          Build info documentation"
	@echo "  $(YELLOW)gettext$(NC)       Build gettext documentation"
	@echo "  $(YELLOW)changes$(NC)       Build changes documentation"
	@echo "  $(YELLOW)xml$(NC)           Build XML documentation"
	@echo "  $(YELLOW)pseudoxml$(NC)     Build pseudo-XML documentation"
	@echo "  $(YELLOW)linkcheck$(NC)     Check all external links"
	@echo "  $(YELLOW)doctest$(NC)       Run doctests in documentation"
	@echo "  $(YELLOW)coverage$(NC)      Check documentation coverage"
	@echo ""
	@echo "$(GREEN)Custom targets:$(NC)"
	@echo "  $(YELLOW)clean$(NC)         Clean build directory"
	@echo "  $(YELLOW)clean-all$(NC)     Clean build directory and API docs"
	@echo "  $(YELLOW)api$(NC)           Generate API documentation"
	@echo "  $(YELLOW)serve$(NC)         Serve documentation locally"
	@echo "  $(YELLOW)install-deps$(NC)  Install documentation dependencies"
	@echo "  $(YELLOW)check-deps$(NC)    Check if dependencies are installed"
	@echo "  $(YELLOW)rebuild$(NC)       Clean and rebuild HTML documentation"

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
	$(PIP) install sphinx sphinx_rtd_theme myst-parser sphinx-autoapi
	@echo "$(GREEN)Documentation dependencies installed$(NC)"

# =============================================================================
# API Documentation
# =============================================================================

.PHONY: api
api: check-sphinx ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(NC)"
	@if [ -d "$(PACKAGE_DIR)" ]; then \
		$(SPHINXAPIDOC) -o $(SOURCEDIR)/api $(PACKAGE_DIR) --force --module-first; \
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
html: check-sphinx ## Build HTML documentation
	@echo "$(BLUE)Building HTML documentation...$(NC)"
	@mkdir -p $(BUILDDIR)
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "$(GREEN)HTML documentation built in $(BUILDDIR)/html$(NC)"
	@echo "$(YELLOW)Open $(BUILDDIR)/html/index.html in your browser$(NC)"

.PHONY: rebuild
rebuild: clean-all api html ## Clean and rebuild HTML documentation
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
	$(SPHINXBUILD) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml $(SPHINXOPTS)
	@echo "$(GREEN)dirhtml documentation built in $(BUILDDIR)/dirhtml$(NC)"

.PHONY: singlehtml
singlehtml: check-sphinx ## Build single HTML page
	@echo "$(BLUE)Building single HTML documentation...$(NC)"
	$(SPHINXBUILD) -b singlehtml $(SOURCEDIR) $(BUILDDIR)/singlehtml $(SPHINXOPTS)
	@echo "$(GREEN)Single HTML documentation built in $(BUILDDIR)/singlehtml$(NC)"

.PHONY: pickle
pickle: check-sphinx ## Build pickle documentation
	$(SPHINXBUILD) -b pickle $(SOURCEDIR) $(BUILDDIR)/pickle $(SPHINXOPTS)

.PHONY: json
json: check-sphinx ## Build JSON documentation
	$(SPHINXBUILD) -b json $(SOURCEDIR) $(BUILDDIR)/json $(SPHINXOPTS)

.PHONY: htmlhelp
htmlhelp: check-sphinx ## Build HTML help documentation
	$(SPHINXBUILD) -b htmlhelp $(SOURCEDIR) $(BUILDDIR)/htmlhelp $(SPHINXOPTS)

.PHONY: qthelp
qthelp: check-sphinx ## Build Qt help documentation
	$(SPHINXBUILD) -b qthelp $(SOURCEDIR) $(BUILDDIR)/qthelp $(SPHINXOPTS)

.PHONY: devhelp
devhelp: check-sphinx ## Build DevHelp documentation
	$(SPHINXBUILD) -b devhelp $(SOURCEDIR) $(BUILDDIR)/devhelp $(SPHINXOPTS)

.PHONY: epub
epub: check-sphinx ## Build EPUB documentation
	@echo "$(BLUE)Building EPUB documentation...$(NC)"
	$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub $(SPHINXOPTS)
	@echo "$(GREEN)EPUB documentation built in $(BUILDDIR)/epub$(NC)"

.PHONY: latex
latex: check-sphinx ## Build LaTeX documentation
	@echo "$(BLUE)Building LaTeX documentation...$(NC)"
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
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
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	cd $(BUILDDIR)/latex && make all-pdf-ja

.PHONY: text
text: check-sphinx ## Build plain text documentation
	$(SPHINXBUILD) -b text $(SOURCEDIR) $(BUILDDIR)/text $(SPHINXOPTS)

.PHONY: man
man: check-sphinx ## Build manual pages
	$(SPHINXBUILD) -b man $(SOURCEDIR) $(BUILDDIR)/man $(SPHINXOPTS)

.PHONY: texinfo
texinfo: check-sphinx ## Build Texinfo documentation
	$(SPHINXBUILD) -b texinfo $(SOURCEDIR) $(BUILDDIR)/texinfo $(SPHINXOPTS)

.PHONY: info
info: texinfo ## Build info documentation
	cd $(BUILDDIR)/texinfo && make info

.PHONY: gettext
gettext: check-sphinx ## Build gettext documentation
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/locale $(SPHINXOPTS)

.PHONY: changes
changes: check-sphinx ## Build changes documentation
	$(SPHINXBUILD) -b changes $(SOURCEDIR) $(BUILDDIR)/changes $(SPHINXOPTS)

.PHONY: xml
xml: check-sphinx ## Build XML documentation
	$(SPHINXBUILD) -b xml $(SOURCEDIR) $(BUILDDIR)/xml $(SPHINXOPTS)

.PHONY: pseudoxml
pseudoxml: check-sphinx ## Build pseudo-XML documentation
	$(SPHINXBUILD) -b pseudoxml $(SOURCEDIR) $(BUILDDIR)/pseudoxml $(SPHINXOPTS)

# =============================================================================
# Quality assurance targets
# =============================================================================

.PHONY: linkcheck
linkcheck: check-sphinx ## Check all external links
	@echo "$(BLUE)Checking external links...$(NC)"
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)
	@echo "$(GREEN)Link check completed. See $(BUILDDIR)/linkcheck/output.txt$(NC)"

.PHONY: doctest
doctest: check-sphinx ## Run doctests in documentation
	@echo "$(BLUE)Running doctests...$(NC)"
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS)
	@echo "$(GREEN)Doctests completed$(NC)"

.PHONY: coverage
coverage: check-sphinx ## Check documentation coverage
	@echo "$(BLUE)Checking documentation coverage...$(NC)"
	$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage $(SPHINXOPTS)
	@echo "$(GREEN)Coverage check completed. See $(BUILDDIR)/coverage/$(NC)"

# =============================================================================
# Development targets
# =============================================================================

.PHONY: watch
watch: html ## Watch for changes and rebuild (requires entr)
	@echo "$(BLUE)Watching for changes...$(NC)"
	@command -v entr >/dev/null 2>&1 || { \
		echo "$(RED)Error: entr not found. Install with: apt install entr$(NC)"; \
		exit 1; \
	}
	find $(SOURCEDIR) -name "*.rst" -o -name "*.md" -o -name "*.py" | entr make html

.PHONY: live
live: ## Live reload server (requires sphinx-autobuild)
	@echo "$(BLUE)Starting live reload server...$(NC)"
	@$(PYTHON) -c "import sphinx_autobuild" 2>/dev/null || { \
		echo "$(RED)Error: sphinx-autobuild not found.$(NC)"; \
		echo "$(YELLOW)Install with: $(PIP) install sphinx-autobuild$(NC)"; \
		exit 1; \
	}
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) --host 0.0.0.0 --port 8000

# =============================================================================
# Default and special targets
# =============================================================================

# Default target
.DEFAULT_GOAL := help

# Catch-all target for any unmatched targets
%: Makefile check-sphinx
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)