# Homepage Motion Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sloppy offset homepage motion collage with the approved Wide lead, calibrated pair while preserving all scientific content and adding static fallbacks.

**Architecture:** Keep the three current homepage subjects and semantic figures, but make the workspace figure span the grid and place the trajectory/pick studies in one aligned 7/5 row. Use stable framed `<picture>` surfaces with contained media, explicit dimensions, and reduced-motion PNG sources. Extend the existing docs design contract and validate the result in real Chromium.

**Tech Stack:** Sphinx/reStructuredText, semantic HTML `<figure>/<picture>`, CSS Grid, Pillow, ImageMagick for one-time fallback extraction, pytest, Chromium.

## Global Constraints

- Preserve the section heading “See the math move” and the three existing scientific subjects.
- Use the approved Wide lead, calibrated pair: full-grid workspace lead over a 7fr/5fr supporting row.
- Never crop axes, legends, tick labels, or robot geometry; use contained media.
- Remove all one-off figure offsets and the old primary two-row span.
- Do not add new colors, shadows, JavaScript, decorative motion, or overlaid labels.
- At widths below 768px, stack all figures in narrative order with no offsets or horizontal overflow.
- Normal HTML may animate; reduced-motion HTML, ePub, and LaTeX/PDF must have static PNGs.
- Preserve unrelated homepage sections, routes, copy, legal text, navigation, and user worktree files.

## File Structure

- Modify `docs/source/index.rst`: accessible picture markup and builder fallbacks.
- Modify `docs/source/_static/custom.css`: wide lead, aligned pair, containment, and responsive behavior.
- Create `docs/source/_static/images/workspace-still.png`: static workspace frame.
- Create `docs/source/_static/images/joint-trajectory-still.png`: static trajectory frame.
- Create `docs/source/_static/images/ur5-pick-motion-still.png`: static pick-motion frame.
- Modify `tests/test_docs_design.py`: source, CSS, image, and builder contracts.

---

### Task 1: Lock the approved gallery contract

**Files:**
- Modify: `tests/test_docs_design.py`

**Interfaces:**
- Produces: regression contract for `.mp-motion-gallery__primary`, picture sources, contained rendering, 7/5 columns, and removal of offsets.

- [ ] **Step 1: Add failing source and CSS tests**

Add:

```python
def test_motion_gallery_uses_wide_lead_calibrated_pair():
    source = read(INDEX)
    css = read(CSS)
    assert 'class="mp-motion-gallery__primary"' in source
    assert re.search(
        r"\.mp-motion-gallery__grid\s*\{[^}]*"
        r"grid-template-columns:\s*minmax\(0,\s*7fr\)\s+minmax\(0,\s*5fr\);",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"\.mp-motion-gallery__primary\s*\{[^}]*grid-column:\s*1\s*/\s*-1;",
        css,
        re.DOTALL,
    )
    assert "grid-row: 1 / span 2" not in css
    gallery_css = css[css.index(".mp-motion-gallery {") : css.index("/* Backend heading")]
    assert "object-fit: cover" not in gallery_css
    assert "nth-child(2)" not in gallery_css
    assert "nth-child(3)" not in gallery_css


def test_motion_gallery_has_reduced_motion_stills():
    source = read(INDEX)
    for stem in (
        "workspace-still.png",
        "joint-trajectory-still.png",
        "ur5-pick-motion-still.png",
    ):
        assert (
            f'<source media="(prefers-reduced-motion: reduce)" '
            f'srcset="_static/images/{stem}">' in source
        )
    assert source.count('class="mp-motion-gallery__media"') == 3
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m pytest tests/test_docs_design.py -k motion_gallery -v`

Expected: FAIL on the current row-spanning layout, cover crop, offsets, and absent still sources.

- [ ] **Step 3: Commit the red contract**

Do not commit failing tests separately. Keep them unstaged until Tasks 2 and 3 make the complete gallery contract pass.

---

### Task 2: Create deterministic static fallbacks

**Files:**
- Create: `docs/source/_static/images/workspace-still.png`
- Create: `docs/source/_static/images/joint-trajectory-still.png`
- Create: `docs/source/_static/images/ur5-pick-motion-still.png`
- Modify: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: existing `_static/gifs/workspace.gif`, `joint_trajectory.gif`, and `ur5_pick_motion.gif`.
- Produces: valid PNG first-frame fallbacks at intrinsic GIF canvas dimensions.

- [ ] **Step 1: Add failing intrinsic-size tests**

Extend `test_static_brand_assets_are_local_and_valid()` or add a focused test:

```python
def test_motion_gallery_static_fallbacks_are_valid():
    expected = {
        "workspace-still.png": (550, 450),
        "joint-trajectory-still.png": (700, 320),
        "ur5-pick-motion-still.png": (480, 360),
    }
    for name, size in expected.items():
        with Image.open(DOCS / "_static" / "images" / name) as image:
            assert image.format == "PNG"
            assert image.size == size
```

- [ ] **Step 2: Verify the absent-asset failure**

Run: `python3 -m pytest tests/test_docs_design.py::test_motion_gallery_static_fallbacks_are_valid -v`

Expected: FAIL because the PNG files do not exist.

- [ ] **Step 3: Extract composited representative frames**

Use ImageMagick's coalesced first frame so disposal metadata cannot create a partial image:

```bash
convert docs/source/_static/gifs/workspace.gif -coalesce -delete 1--1 docs/source/_static/images/workspace-still.png
convert docs/source/_static/gifs/joint_trajectory.gif -coalesce -delete 1--1 docs/source/_static/images/joint-trajectory-still.png
convert docs/source/_static/gifs/ur5_pick_motion.gif -coalesce -delete 1--1 docs/source/_static/images/ur5-pick-motion-still.png
```

Inspect all three PNGs and confirm the plot axes/labels and robot are visible. If an animation's first frame is intentionally blank, select the earliest complete coalesced frame by explicit zero-based index and record that exact index in the commit message body.

- [ ] **Step 4: Run the image contract**

Run: `python3 -m pytest tests/test_docs_design.py::test_motion_gallery_static_fallbacks_are_valid -v`

Expected: PASS.

---

### Task 3: Implement the wide lead and calibrated pair

**Files:**
- Modify: `docs/source/index.rst`
- Modify: `docs/source/_static/custom.css`
- Modify: `tests/test_docs_design.py`

**Interfaces:**
- Consumes: Task 2 PNG fallbacks and the three existing GIFs.
- Produces: one full-grid primary figure and one aligned 7/5 supporting row.

- [ ] **Step 1: Replace each raw image with a stable picture surface**

Keep the existing `<figure>` order and captions. Inside each figure use:

```html
<picture class="mp-motion-gallery__media">
   <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/workspace-still.png">
   <img src="_static/gifs/workspace.gif" width="550" height="450" loading="lazy" alt="Robot arm tracing its reachable workspace">
</picture>
```

Use these other two exact media blocks:

```html
<picture class="mp-motion-gallery__media">
   <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/joint-trajectory-still.png">
   <img src="_static/gifs/joint_trajectory.gif" width="700" height="320" loading="lazy" alt="Joint position, velocity, and acceleration changing along a trajectory">
</picture>
<picture class="mp-motion-gallery__media">
   <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/ur5-pick-motion-still.png">
   <img src="_static/gifs/ur5_pick_motion.gif" width="480" height="360" loading="lazy" alt="UR5 robot executing a pick motion">
</picture>
```

Keep `.mp-motion-gallery__primary` only on the workspace figure. Add ePub and LaTeX static figure branches after the HTML gallery using the three PNGs and the captions “Explore reachable workspaces,” “Inspect a smooth joint trajectory,” and “Execute planned motion in simulation.” Preserve the existing concise LaTeX explanatory text after those figures.

- [ ] **Step 2: Replace the gallery grid CSS**

Use this structural contract:

```css
.mp-motion-gallery__grid {
  display: grid;
  grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
  gap: clamp(1.5rem, 3vw, 3rem) clamp(1.5rem, 4vw, 4rem);
  align-items: stretch;
}

.mp-motion-gallery figure {
  display: flex;
  min-width: 0;
  margin: 0;
  flex-direction: column;
}

.mp-motion-gallery__primary {
  grid-column: 1 / -1;
}

.mp-motion-gallery__media {
  display: grid;
  min-height: 0;
  flex: 1;
  place-items: center;
  overflow: hidden;
  background: var(--mp-panel-strong);
  border: 1px solid var(--mp-rule);
  border-radius: var(--mp-radius);
}

.mp-motion-gallery__primary .mp-motion-gallery__media {
  block-size: clamp(24rem, 48vw, 38rem);
}

.mp-motion-gallery__grid > figure:not(.mp-motion-gallery__primary) .mp-motion-gallery__media {
  block-size: clamp(16rem, 28vw, 23rem);
}

.mp-motion-gallery img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
```

Retain the established caption typography. Remove the `nth-child` margins, primary row span, and all gallery `object-fit: cover` rules.

- [ ] **Step 3: Normalize the mobile rule**

Inside the existing `@media (max-width: 767px)` block, keep one column and set both primary/support media to `block-size: auto; aspect-ratio: 4 / 3;`. Keep `object-fit: contain`; do not restore offsets.

- [ ] **Step 4: Run the complete gallery contract**

Run: `python3 -m pytest tests/test_docs_design.py -k 'motion_gallery or static_fallbacks' -v`

Expected: all gallery tests pass.

- [ ] **Step 5: Build HTML and inspect generated markup**

Run: `make -C docs html SPHINXSTRICT=1`

Expected: no new warnings, three GIF `<img>` elements, three reduced-motion PNG sources, and the static builder branches excluded from ordinary HTML.

- [ ] **Step 6: Commit the complete gallery correction**

```bash
git add docs/source/index.rst docs/source/_static/custom.css docs/source/_static/images/workspace-still.png docs/source/_static/images/joint-trajectory-still.png docs/source/_static/images/ur5-pick-motion-still.png tests/test_docs_design.py
git commit -m "fix: align the homepage motion gallery"
```

---

### Task 4: Real-browser and multi-builder verification

**Files:**
- Modify only Task 2-3 files if a verified gallery defect requires correction.

**Interfaces:**
- Consumes: completed gallery and test contract.
- Produces: evidence that composition, containment, motion preference, and static formats work.

- [ ] **Step 1: Run regression tests**

Run: `python3 -m pytest tests/test_docs_design.py tests/test_docs_tutorials.py -v`

Expected: all tests pass.

- [ ] **Step 2: Build HTML, ePub, and LaTeX**

Run: `make -C docs html SPHINXSTRICT=1`

Run: `make -C docs epub SPHINXSTRICT=1`

Run: `make -C docs latex SPHINXSTRICT=1`

Expected: all builders complete without new warnings; ePub/LaTeX outputs reference the three PNG fallbacks rather than gallery GIFs.

- [ ] **Step 3: Inspect desktop composition in Chromium**

Serve `docs/build/html` and inspect the homepage at 1440, 1200, 960, 768, 390, and 320 CSS pixels in light and dark themes. At desktop widths verify:

- workspace figure spans both columns;
- supporting figures use a 7/5 row and their media bottoms/captions align;
- every axis, legend, tick label, and robot remains visible;
- matte space is deliberate and uses the panel token;
- there is no horizontal page overflow.

- [ ] **Step 4: Inspect reduced motion and mobile**

Emulate `prefers-reduced-motion: reduce` and verify each `<picture>` selects its PNG source and reveal animation is disabled. At 390 and 320 widths verify one-column narrative order, identical side gutters, contained 4/3 media, readable captions, and no one-off gaps.

- [ ] **Step 5: Apply the design-taste pre-flight**

Check that the section has one dominant visual thesis, one supporting pair, no floating rounded cards, no excessive labels, no arbitrary offsets, no cropped scientific content, and no animation beyond the source studies and existing reveal behavior.

- [ ] **Step 6: Run final repository checks**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors; only explicitly preserved pre-existing untracked files may remain.

- [ ] **Step 7: Commit verified corrections if any**

If browser or builder QA exposed an in-scope defect, stage only the corrected gallery files and commit:

```bash
git commit -m "fix: complete motion gallery verification"
```

If no corrections were needed, do not create an empty commit.
