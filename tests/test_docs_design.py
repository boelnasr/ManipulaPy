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
