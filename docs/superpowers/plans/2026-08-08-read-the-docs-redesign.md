# ManipulaPy Documentation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic ManipulaPy documentation presentation with the approved Editorial Robotics Lab design while preserving Sphinx, PyData, every published route, and all documentation behavior.

**Architecture:** Keep `pydata-sphinx-theme` as the accessible documentation shell and implement the redesign through one concise reStructuredText homepage, one semantic CSS system, and one dependency-free progressive motion script. Vendor the two approved font files and one generated hero image under Sphinx static assets so Read the Docs builds remain deterministic and offline-friendly.

**Tech Stack:** Sphinx 8, pydata-sphinx-theme 0.18-0.19, reStructuredText with constrained raw HTML, CSS custom properties and Grid, native JavaScript `IntersectionObserver` and Web Animations API, pytest, Pillow, headless Chrome.

## Global Constraints

- Preserve every route slug, primary navigation label, search behavior, theme switching behavior, source link, left navigation, and right table of contents.
- Use `DESIGN_VARIANCE: 7`, `MOTION_INTENSITY: 4`, and `VISUAL_DENSITY: 5`.
- Use a light-first cold-neutral palette with one teal accent and a complete dark-mode equivalent.
- Use self-hosted Space Grotesk headings, system-sans prose, and self-hosted JetBrains Mono code.
- Use one consistent 6px corner radius and no glow, glass, decorative status dots, overlaid image labels, fake interfaces, or three-equal-card layouts.
- Visible homepage copy must contain zero em-dash or en-dash characters.
- The hero headline is "Move from equations to motion." The supporting sentence is no more than 20 words. The only primary CTA is "Start building."
- Keep JavaScript optional, honor `prefers-reduced-motion`, and animate only transforms and opacity.
- Do not modify Python package behavior, API names, Read the Docs deployment targets, legal text, or unrelated dirty worktree files.

## File Map

- `tests/test_docs_design.py`: source-level contract for assets, homepage structure, accessibility hooks, CSS tokens, reduced motion, and prohibited patterns.
- `docs/source/conf.py`: one-time static asset registration, concise brand title, and PyData navigation configuration.
- `docs/source/index.rst`: homepage information architecture and semantic content.
- `docs/source/_static/custom.css`: complete Editorial Robotics Lab token and component system for home, guide, API, mobile, light, and dark contexts.
- `docs/source/_static/motion.js`: optional one-time reveal behavior with no third-party runtime.
- `docs/source/_static/images/robotics-lab-hero.webp`: generated hero image at 1600 by 1100 pixels.
- `docs/source/_static/fonts/SpaceGrotesk-Variable.ttf`: Space Grotesk variable font from the official Google Fonts repository.
- `docs/source/_static/fonts/JetBrainsMono-Variable.woff2`: JetBrains Mono variable font from the official JetBrains repository.
- `docs/source/_static/fonts/OFL-Space-Grotesk.txt` and `docs/source/_static/fonts/OFL-JetBrains-Mono.txt`: upstream font licenses.
- Remove `docs/source/_static/anime.umd.min.js` after the native motion script is verified.

---

### Task 1: Lock the Documentation Design Contract

**Files:**
- Create: `tests/test_docs_design.py`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: approved specification at `docs/superpowers/specs/2026-08-08-read-the-docs-redesign-design.md`.
- Produces: source-level assertions that every later task must satisfy.

- [ ] **Step 1: Add the failing design-contract tests**

```python
from pathlib import Path
import re

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
INDEX = DOCS / "index.rst"
CSS = DOCS / "_static" / "custom.css"
MOTION = DOCS / "_static" / "motion.js"
CONF = DOCS / "conf.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_homepage_has_approved_sections_and_copy():
    source = read(INDEX)
    for class_name in (
        "mp-home",
        "mp-hero",
        "mp-paths",
        "mp-motion-gallery",
        "mp-backends",
        "mp-api-links",
    ):
        assert class_name in source
    assert "Move from equations to motion." in source
    assert source.count("Start building") == 1


def test_homepage_removes_clutter_and_forbidden_copy():
    source = read(INDEX)
    assert "mp-badges" not in source
    assert ".. contents::" not in source
    assert "Worked examples" not in source
    assert "Measured performance" not in source
    assert "Release notes" not in source
    assert "—" not in source
    assert "–" not in source


def test_homepage_preserves_navigation_documents():
    source = read(INDEX)
    for document in (
        "Installation Guide",
        "getting_started/index",
        "tutorials/index",
        "api/index",
        "user_guide/index",
    ):
        assert document in source


def test_static_brand_assets_are_local_and_valid():
    fonts = DOCS / "_static" / "fonts"
    assert (fonts / "SpaceGrotesk-Variable.ttf").stat().st_size > 50_000
    assert (fonts / "JetBrainsMono-Variable.woff2").stat().st_size > 50_000
    assert (fonts / "OFL-Space-Grotesk.txt").is_file()
    assert (fonts / "OFL-JetBrains-Mono.txt").is_file()
    hero = DOCS / "_static" / "images" / "robotics-lab-hero.webp"
    with Image.open(hero) as image:
        assert image.format == "WEBP"
        assert image.size == (1600, 1100)


def test_theme_assets_are_registered_once():
    conf = read(CONF)
    assert 'html_css_files = ["custom.css"]' in conf
    assert 'html_js_files = ["motion.js"]' in conf
    assert conf.count('add_css_file("custom.css")') == 0
    assert "anime.umd.min.js" not in conf


def test_css_has_theme_responsive_and_accessibility_contracts():
    css = read(CSS)
    for token in (
        "--mp-accent",
        "--mp-canvas",
        "--mp-panel",
        "--mp-ink",
        "--mp-radius: 6px",
    ):
        assert token in css
    assert 'html[data-theme="dark"]' in css
    assert "@media (max-width: 767px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ":focus-visible" in css
    assert "min-height: 100dvh" in css
    assert "#000000" not in css.lower()
    assert "#ffffff" not in css.lower()


def test_motion_is_progressive_and_scroll_listener_free():
    script = read(MOTION)
    assert "IntersectionObserver" in script
    assert "prefers-reduced-motion: reduce" in script
    assert ".animate(" in script
    assert 'addEventListener("scroll"' not in script
    assert "anime" not in script.lower()


def test_eyebrow_budget_is_not_exceeded():
    source = read(INDEX)
    section_count = len(re.findall(r'<section class="mp-(?:hero|paths|motion-gallery|backends|api-links)"', source))
    eyebrow_count = source.count('class="mp-overline"')
    assert section_count == 5
    assert eyebrow_count <= 2
```

- [ ] **Step 2: Run the contract to prove the current design fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q`

Expected: failures for missing homepage classes, missing local fonts and hero asset, old Anime.js registration, and missing new CSS tokens.

- [ ] **Step 3: Commit the test contract**

```bash
git add tests/test_docs_design.py
git commit -m "test: define documentation design contract"
```

---

### Task 2: Vendor Brand Assets and Simplify Theme Configuration

**Files:**
- Create: `docs/source/_static/fonts/SpaceGrotesk-Variable.ttf`
- Create: `docs/source/_static/fonts/JetBrainsMono-Variable.woff2`
- Create: `docs/source/_static/fonts/OFL-Space-Grotesk.txt`
- Create: `docs/source/_static/fonts/OFL-JetBrains-Mono.txt`
- Create: `docs/source/_static/images/robotics-lab-hero.webp`
- Modify: `docs/source/conf.py`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: exact asset paths asserted by the design contract.
- Produces: deterministic local typography, one 1600 by 1100 WEBP hero asset, and exactly one CSS and one JavaScript registration.

- [ ] **Step 1: Download the pinned upstream font files and licenses**

```bash
mkdir -p docs/source/_static/fonts docs/source/_static/images
curl -L "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf" -o docs/source/_static/fonts/SpaceGrotesk-Variable.ttf
curl -L "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/OFL.txt" -o docs/source/_static/fonts/OFL-Space-Grotesk.txt
curl -L "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/fonts/webfonts/JetBrainsMono%5Bwght%5D.woff2" -o docs/source/_static/fonts/JetBrainsMono-Variable.woff2
curl -L "https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/OFL.txt" -o docs/source/_static/fonts/OFL-JetBrains-Mono.txt
```

Expected: both font files exceed 50 KB and both license files contain the SIL Open Font License text.

- [ ] **Step 2: Generate and normalize the hero asset**

Use the image-generation tool with this exact art direction:

```text
Create a 1600x1100 editorial hero image for ManipulaPy robotics documentation. A precise six-axis industrial robot arm in a clean research lab, captured from a low three-quarter angle. Cold silver and graphite materials, one restrained teal light accent, soft daylight, generous negative space on the left for adjacent web typography, technically credible joints and end effector, no people, no text, no logos, no interface overlays, no neon glow, no purple, no sci-fi control room. Premium technical publication photography, crisp but not sterile.
```

Save the generated result as a temporary source image, then normalize it with Pillow:

```python
from pathlib import Path
from PIL import Image, ImageOps

source = Path("/tmp/manipulapy-hero-source.png")
target = Path("docs/source/_static/images/robotics-lab-hero.webp")
with Image.open(source) as image:
    image = ImageOps.fit(image.convert("RGB"), (1600, 1100), method=Image.Resampling.LANCZOS)
    image.save(target, "WEBP", quality=88, method=6)
```

- [ ] **Step 3: Simplify Sphinx theme registration**

In `docs/source/conf.py`:

- Keep `pydata_sphinx_theme` as the preferred theme and `sphinx_rtd_theme` as the fallback.
- Set `html_title = "ManipulaPy Documentation"` and `html_short_title = "ManipulaPy"`.
- Set `html_css_files = ["custom.css"]` and `html_js_files = ["motion.js"]`.
- Remove Anime.js from `html_js_files`.
- Remove `app.add_css_file("custom.css")` from `setup()` so the stylesheet is not registered twice.
- Keep search, theme switcher, GitHub, PyPI, and the existing navigation-depth settings.

- [ ] **Step 4: Run the asset and configuration contract tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q`

Expected: asset and theme-registration tests pass; homepage, CSS, and motion tests still fail.

- [ ] **Step 5: Commit the local assets and configuration**

```bash
git add docs/source/conf.py docs/source/_static/fonts docs/source/_static/images/robotics-lab-hero.webp
git commit -m "docs: add local editorial brand assets"
```

---

### Task 3: Recompose the Documentation Homepage

**Files:**
- Modify: `docs/source/index.rst`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: hero image at `_static/images/robotics-lab-hero.webp`, existing GIFs under `_static/gifs`, and unchanged Sphinx document routes.
- Produces: five semantic homepage sections and hidden toctrees that continue to drive navigation.

- [ ] **Step 1: Replace the long homepage with the approved five-section structure**

Use this outer structure and exact primary copy. Keep the raw HTML tags balanced and use root-relative documentation links matching the current generated routes.

```rst
.. _doc-index:

ManipulaPy Documentation
========================

.. raw:: html

   <div class="mp-home">
      <section class="mp-hero" aria-labelledby="mp-hero-title">
         <div class="mp-hero__copy">
            <p class="mp-overline">Python robotics, from model to motion</p>
            <h1 id="mp-hero-title">Move from equations to motion.</h1>
            <p class="mp-hero__lede">Build, differentiate, and accelerate robot kinematics, dynamics, planning, perception, and control in Python.</p>
            <a class="mp-primary-action" href="getting_started/index.html">Start building</a>
            <div class="mp-install" aria-label="Installation command">
               <code>python -m pip install manipulapy</code>
            </div>
         </div>
         <figure class="mp-hero__media">
            <img src="_static/images/robotics-lab-hero.webp" width="1600" height="1100" alt="Six-axis industrial robot arm in a research lab">
         </figure>
      </section>

      <section class="mp-paths" aria-labelledby="mp-paths-title" data-reveal>
         <h2 id="mp-paths-title">Choose your path</h2>
         <div class="mp-paths__grid">
            <a class="mp-path mp-path--featured" href="tutorials/notebook_course.html">
               <span>Learn robotics</span>
               <h3>Work through the mathematics</h3>
               <p>Follow executable notebooks from rigid transforms through differentiable dynamics.</p>
            </a>
            <a class="mp-path" href="user_guide/URDF_Processor.html">
               <span>Load a robot</span>
               <h3>Start from a URDF</h3>
               <p>Build the serial model, dynamics, limits, and frames from your robot description.</p>
            </a>
            <a class="mp-path" href="user_guide/Trajectory_Planning.html">
               <span>Plan motion</span>
               <h3>Turn goals into trajectories</h3>
               <p>Generate joint and Cartesian paths, then connect planning to control and simulation.</p>
            </a>
            <a class="mp-path" href="user_guide/Backends.html">
               <span>Accelerate and differentiate</span>
               <h3>Choose the right array backend</h3>
               <p>Run the same core mathematics on NumPy, CuPy, PyTorch, or JAX.</p>
            </a>
         </div>
      </section>

      <section class="mp-motion-gallery" aria-labelledby="mp-motion-title" data-reveal>
         <h2 id="mp-motion-title">See the math move</h2>
         <div class="mp-motion-gallery__grid">
            <figure class="mp-motion-gallery__primary">
               <img src="_static/gifs/workspace.gif" width="550" height="450" loading="lazy" alt="Robot arm tracing its reachable workspace">
               <figcaption>Explore reachable workspaces</figcaption>
            </figure>
            <figure>
               <img src="_static/gifs/joint_trajectory.gif" width="700" height="320" loading="lazy" alt="Joint position, velocity, and acceleration changing along a trajectory">
               <figcaption>Inspect a smooth joint trajectory</figcaption>
            </figure>
            <figure>
               <img src="_static/gifs/ur5_pick_motion.gif" width="480" height="360" loading="lazy" alt="UR5 robot executing a pick motion">
               <figcaption>Execute planned motion in simulation</figcaption>
            </figure>
         </div>
      </section>

      <section class="mp-backends" aria-labelledby="mp-backends-title" data-reveal>
         <p class="mp-overline">One API, four array libraries</p>
         <h2 id="mp-backends-title">Compute where your work belongs.</h2>
         <div class="mp-backends__grid">
            <div><h3>NumPy</h3><p>The lightweight default for local work, teaching, and dependable CPU execution.</p></div>
            <div><h3>CuPy</h3><p>Move compatible array work to NVIDIA GPUs without changing the public mathematics API.</p></div>
            <div><h3>PyTorch</h3><p>Connect robot models to training loops and preserve gradients through supported core math.</p></div>
            <div><h3>JAX</h3><p>Differentiate and compile supported kinematics, dynamics, singularity, and utility operations.</p></div>
         </div>
         <div class="mp-backends__links">
            <a href="user_guide/Backends.html">Read the backend guide</a>
            <a href="api/backend.html">Open the backend API</a>
         </div>
      </section>

      <section class="mp-api-links" aria-labelledby="mp-api-title" data-reveal>
         <h2 id="mp-api-title">Go straight to the reference</h2>
         <div class="mp-api-links__grid">
            <div><h3>Model</h3><a href="api/kinematics.html">Kinematics</a><a href="api/dynamics.html">Dynamics</a><a href="api/urdf_processor.html">URDF processor</a></div>
            <div><h3>Move</h3><a href="api/path_planning.html">Path planning</a><a href="api/control.html">Control</a><a href="api/potential_field.html">Potential fields</a></div>
            <div><h3>Sense</h3><a href="api/vision.html">Vision</a><a href="api/perception.html">Perception</a><a href="api/simulation.html">Simulation</a></div>
            <div><h3>Compute</h3><a href="api/backend.html">Backends</a><a href="api/cuda_kernels.html">CUDA kernels</a><a href="api/utils.html">Utilities</a></div>
         </div>
      </section>
   </div>
```

Fill the learning paths with existing destinations:

- Learn robotics: `tutorials/notebook_course.html`
- Load a robot: `user_guide/URDF_Processor.html`
- Plan motion: `user_guide/Trajectory_Planning.html`
- Accelerate and differentiate: `user_guide/Backends.html`

Group API links without changing route names:

- Model: `api/kinematics.html`, `api/dynamics.html`, `api/urdf_processor.html`
- Move: `api/path_planning.html`, `api/control.html`, `api/potential_field.html`
- Sense: `api/vision.html`, `api/perception.html`, `api/simulation.html`
- Compute: `api/backend.html`, `api/cuda_kernels.html`, `api/utils.html`

- [ ] **Step 2: Preserve navigation through hidden toctrees**

Append these unchanged document roots after the raw homepage markup:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   Installation Guide
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   user_guide/index
```

- [ ] **Step 3: Run homepage contract and RST parser checks**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q`

Expected: homepage structure, route preservation, CTA, clutter-removal, dash, and eyebrow-budget tests pass; CSS and motion tests still fail.

Run: `python3 -m sphinx -b dummy docs/source /tmp/manipulapy-docs-dummy -W --keep-going`

Expected: no new malformed-HTML, unknown-document, or toctree warnings attributable to `index.rst`.

- [ ] **Step 4: Commit the homepage composition**

```bash
git add docs/source/index.rst
git commit -m "docs: recompose the documentation homepage"
```

---

### Task 4: Implement the Editorial Robotics Lab Design System

**Files:**
- Modify: `docs/source/_static/custom.css`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: the six homepage class families and nested elements from Task 3, plus PyData's stable `bd-*` shell classes.
- Produces: light and dark semantic tokens, responsive homepage layouts, and refined guide/API components.

- [ ] **Step 1: Replace the existing stylesheet with a token-first system**

Start with these exact foundations:

```css
@font-face {
  font-family: "Space Grotesk";
  src: url("fonts/SpaceGrotesk-Variable.ttf") format("truetype");
  font-weight: 300 700;
  font-style: normal;
  font-display: swap;
}

@font-face {
  font-family: "JetBrains Mono";
  src: url("fonts/JetBrainsMono-Variable.woff2") format("woff2");
  font-weight: 100 800;
  font-style: normal;
  font-display: swap;
}

html[data-theme="light"] {
  --mp-canvas: #f2f4f5;
  --mp-panel: #e7ebed;
  --mp-panel-strong: #d8dfe2;
  --mp-ink: #172126;
  --mp-muted: #53636b;
  --mp-rule: #c4ced2;
  --mp-accent: #087f75;
  --mp-accent-strong: #05645d;
  --mp-code: #10181c;
  --mp-code-ink: #e4ecee;
  --mp-radius: 6px;
}

html[data-theme="dark"] {
  --mp-canvas: #11181d;
  --mp-panel: #182228;
  --mp-panel-strong: #223038;
  --mp-ink: #dce5e8;
  --mp-muted: #9cabb1;
  --mp-rule: #34454d;
  --mp-accent: #63c9bc;
  --mp-accent-strong: #8edfd5;
  --mp-code: #0c1216;
  --mp-code-ink: #dce8e9;
  --mp-radius: 6px;
}

body {
  background: var(--mp-canvas);
  color: var(--mp-ink);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3, h4, h5, h6, .navbar-brand {
  font-family: "Space Grotesk", system-ui, sans-serif;
  letter-spacing: -0.035em;
}

code, pre, kbd, samp, .highlight {
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
```

- [ ] **Step 2: Implement each homepage section as a distinct layout family**

- `.mp-hero`: asymmetric two-column grid using `min-height: 100dvh`, top padding no greater than 6rem, a maximum two-line heading, stable media aspect ratio, and a one-column fallback below 768px.
- `.mp-paths`: asymmetric 7/5 grid with one large learning route and three compact text routes. Do not render four equal cards.
- `.mp-motion-gallery`: one wide media frame plus two vertically offset supporting frames. Use `object-fit: cover`, stable aspect ratios, and no overlaid text.
- `.mp-backends`: a full-width restrained panel with four capability columns separated by sparse vertical rules on desktop and a two-by-two grid on tablet.
- `.mp-api-links`: four grouped link columns using spacing and one group border, not card containers.
- `.mp-overline`: only two occurrences are permitted by the homepage source contract.

- [ ] **Step 3: Reskin the PyData documentation shell**

Cover these selectors without altering DOM behavior:

```css
.bd-header { min-height: 68px; max-height: 72px; }
.navbar-header-items, .navbar-header-items__center { flex-wrap: nowrap; }
.bd-sidebar-primary, .bd-sidebar-secondary { background: var(--mp-canvas); }
.bd-main .bd-content .bd-article-container { max-width: 76ch; }
.bd-content .highlight, .bd-content pre { border-radius: var(--mp-radius); }
.bd-content .admonition { border-radius: var(--mp-radius); box-shadow: none; }
.bd-content dl.py > dt.sig { border-left: 3px solid var(--mp-accent); }
a:focus-visible, button:focus-visible, input:focus-visible {
  outline: 2px solid var(--mp-accent);
  outline-offset: 3px;
}
```

Use the same palette and radius for search, copy buttons, tables, admonitions, API signatures, navigation links, and theme controls. At 1024px, hide or condense secondary top-navigation labels before allowing a wrap.

- [ ] **Step 4: Add explicit mobile, print, and accessibility fallbacks**

```css
@media (max-width: 767px) {
  .mp-hero,
  .mp-paths,
  .mp-motion-gallery,
  .mp-backends__grid,
  .mp-api-links__grid {
    grid-template-columns: 1fr;
  }

  .mp-home { padding-inline: 1rem; }
  .mp-hero { min-height: auto; padding-block: 4rem 3rem; }
  .mp-hero h1 { font-size: clamp(2.7rem, 13vw, 4rem); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}

@media print {
  .mp-hero { min-height: auto; }
  .mp-hero__media, .mp-motion-gallery { break-inside: avoid; }
}
```

- [ ] **Step 5: Run CSS contract tests and build HTML**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q`

Expected: every test except the native-motion contract passes.

Run: `make -C docs html`

Expected: HTML succeeds with no new CSS asset or static-file warnings.

- [ ] **Step 6: Commit the design system**

```bash
git add docs/source/_static/custom.css
git commit -m "style: add editorial robotics docs system"
```

---

### Task 5: Replace Anime.js With Progressive Native Motion

**Files:**
- Modify: `docs/source/_static/motion.js`
- Delete: `docs/source/_static/anime.umd.min.js`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: `[data-reveal]` sections emitted by Task 3.
- Produces: one-time transform and opacity reveals that never leave content hidden after failure.

- [ ] **Step 1: Replace the current runtime-specific script**

```javascript
(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reducedMotion.matches) return;

  function reveal(element) {
    var animation = element.animate(
      [
        { opacity: 0, transform: "translateY(18px)" },
        { opacity: 1, transform: "translateY(0)" }
      ],
      {
        duration: 520,
        easing: "cubic-bezier(0.16, 1, 0.3, 1)",
        fill: "both"
      }
    );
    animation.addEventListener("finish", function () {
      element.style.opacity = "";
      element.style.transform = "";
    }, { once: true });
  }

  function start() {
    var targets = Array.prototype.slice.call(
      document.querySelectorAll("[data-reveal]")
    );
    if (!targets.length) return;

    if (!("IntersectionObserver" in window)) {
      targets.forEach(reveal);
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        reveal(entry.target);
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });

    targets.forEach(function (target) {
      if (target.getBoundingClientRect().top > window.innerHeight) {
        observer.observe(target);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
```

Do not hide content in CSS. The start state exists only inside each running Web Animation, so blocked or failed JavaScript leaves a complete static page.

- [ ] **Step 2: Remove the unused Anime.js bundle**

Run: `git rm docs/source/_static/anime.umd.min.js`

- [ ] **Step 3: Run the full design contract and HTML build**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q`

Expected: all tests pass.

Run: `make -C docs html`

Expected: HTML succeeds and generated `index.html` references `motion.js` but not `anime.umd.min.js`.

- [ ] **Step 4: Commit native motion**

```bash
git add docs/source/conf.py docs/source/_static/motion.js tests/test_docs_design.py
git add -u docs/source/_static/anime.umd.min.js
git commit -m "refactor: use native docs motion"
```

---

### Task 6: Build, Visual QA, and Pre-Flight

**Files:**
- Modify only if QA finds a concrete defect: `docs/source/index.rst`, `docs/source/_static/custom.css`, `docs/source/_static/motion.js`, or `tests/test_docs_design.py`
- Test: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: completed HTML design and existing Sphinx builder configuration.
- Produces: verified HTML, ePub, and LaTeX/PDF inputs plus desktop/mobile light/dark visual evidence.

- [ ] **Step 1: Run clean source and builder verification**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_docs_design.py -q
make -C docs clean
make -C docs html
make -C docs epub
make -C docs latex
```

Expected: the design contract passes; HTML, ePub, and LaTeX sources build without new warnings. If `latexmk` is available, also run `make -C docs latexpdf` and require `docs/build/latex/manipulapy.pdf` to exist.

- [ ] **Step 2: Verify generated routes and assets**

```bash
test -f docs/build/html/index.html
test -f docs/build/html/getting_started/index.html
test -f docs/build/html/user_guide/Backends.html
test -f docs/build/html/api/kinematics.html
grep -q "robotics-lab-hero.webp" docs/build/html/index.html
grep -q "motion.js" docs/build/html/index.html
! grep -q "anime.umd.min.js" docs/build/html/index.html
```

Expected: every command exits successfully.

- [ ] **Step 3: Serve the built documentation and capture four homepage views**

Run `python3 -m http.server 8765 --directory docs/build/html` in a long-running terminal, then capture:

```bash
google-chrome --headless=new --disable-gpu --hide-scrollbars --window-size=1440,1000 --screenshot=/tmp/manipulapy-docs-light-desktop.png http://127.0.0.1:8765/
google-chrome --headless=new --disable-gpu --hide-scrollbars --force-dark-mode --window-size=1440,1000 --screenshot=/tmp/manipulapy-docs-dark-desktop.png http://127.0.0.1:8765/
google-chrome --headless=new --disable-gpu --hide-scrollbars --window-size=390,844 --screenshot=/tmp/manipulapy-docs-light-mobile.png http://127.0.0.1:8765/
google-chrome --headless=new --disable-gpu --hide-scrollbars --force-dark-mode --window-size=390,844 --screenshot=/tmp/manipulapy-docs-dark-mobile.png http://127.0.0.1:8765/
```

Inspect all four images. Require: one-line desktop navigation, hero CTA visible in the initial desktop viewport, no clipped headline, no horizontal mobile overflow, stable image space, readable code, and consistent accent/radius/theme treatment.

- [ ] **Step 4: Inspect representative inner pages**

Capture and inspect `getting_started/index.html`, `user_guide/Backends.html`, and `api/kinematics.html` at 1440 by 1000 and 390 by 844. Verify search, navigation, right table of contents, API signatures, admonitions, code-copy buttons, and source links remain visible and usable.

- [ ] **Step 5: Run the mechanical pre-flight**

```bash
grep -RInE '—|–' docs/source/index.rst docs/source/_static/custom.css docs/source/_static/motion.js && exit 1 || true
grep -RInE 'h-screen|#000000|#ffffff|addEventListener\(["'"']scroll' docs/source/index.rst docs/source/_static/custom.css docs/source/_static/motion.js && exit 1 || true
git diff --check
git status --short
```

Confirm manually that the homepage contains five layout sections, at most two overlines, one primary CTA intent, three real supporting GIFs, no fake-precise metrics, no image overlays, no decorative captions, and no repeated equal-card section.

- [ ] **Step 6: Commit only verified QA corrections**

If QA required changes, stage only the documentation redesign files and contract test:

```bash
git add docs/source/index.rst docs/source/conf.py docs/source/_static/custom.css docs/source/_static/motion.js tests/test_docs_design.py
git commit -m "fix: polish documentation responsive design"
```

If no tracked files changed during QA, skip this commit.

- [ ] **Step 7: Final scope check**

Run: `git status --short`

Expected: unrelated pre-existing Python and test changes remain untouched; all redesign commits contain only the paths named in this plan.
