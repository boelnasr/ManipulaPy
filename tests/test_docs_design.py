import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
INDEX = DOCS / "index.rst"
CSS = DOCS / "_static" / "custom.css"
MOTION = DOCS / "_static" / "motion.js"
CONF = DOCS / "conf.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def theme_tokens(css: str, theme: str) -> dict[str, str]:
    block = re.search(
        rf'html\[data-theme="{theme}"\]\s*\{{(?P<body>.*?)\}}',
        css,
        re.DOTALL,
    )
    assert block is not None
    return dict(
        re.findall(r"--(?P<name>mp-[\w-]+):\s*(?P<value>#[0-9a-fA-F]{6});", block["body"])
    )


def relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_homepage_has_approved_sections_and_page_toc_headings():
    source = read(INDEX)
    editorial_headings = (
        "Move from equations to motion.",
        "Choose your path",
        "See the math move",
        "Compute where your work belongs.",
        "Go straight to the reference",
    )
    for class_name in (
        "mp-hero",
        "mp-paths",
        "mp-motion-gallery",
        "mp-backends",
        "mp-api-links",
    ):
        assert f".. rst-class:: {class_name}" in source
    assert len(re.findall(r"^.+\n-{3,}$", source, re.MULTILINE)) == 5
    for heading in editorial_headings:
        assert re.search(rf"^{re.escape(heading)}\n-{{3,}}$", source, re.MULTILINE)
    assert '<section class="mp-' not in source
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


def test_homepage_preserves_exact_agpl_notice():
    source = read(INDEX)
    agpl_notice = """ManipulaPy is released under the **AGPL-3.0 License**: the source is freely
available, derivative works must also be open source, modified network services
must offer their source to users, and commercial use is permitted under those
same terms. For commercial licensing options or AGPL compliance questions,
please contact the maintainers."""
    assert agpl_notice in source


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
        "--mp-focus",
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


def test_theme_accent_and_focus_tokens_meet_wcag_contrast_contract():
    css = read(CSS)
    light = theme_tokens(css, "light")
    dark = theme_tokens(css, "dark")

    for surface in ("mp-canvas", "mp-panel", "mp-panel-strong"):
        assert contrast_ratio(light["mp-accent"], light[surface]) >= 4.5

    for surface in ("mp-canvas", "mp-panel", "mp-panel-strong", "mp-code"):
        assert contrast_ratio(light["mp-focus"], light[surface]) >= 3.0
        assert contrast_ratio(dark["mp-focus"], dark[surface]) >= 3.0

    assert ".bd-content a.headerlink" in css
    assert re.search(
        r"\.bd-header button:focus,[^{]+\{[^}]*box-shadow:\s*none\s*!important;",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"\.bd-header button:focus-visible,[^{]+\{[^}]*"
        r"outline:\s*3px solid var\(--mp-focus\);[^}]*"
        r"box-shadow:\s*none\s*!important;",
        css,
        re.DOTALL,
    )
    assert re.search(
        r"\.bd-content div\.highlight button\.copybtn:focus-visible\s*\{[^}]*"
        r"outline:\s*2px solid var\(--mp-focus\);[^}]*"
        r"box-shadow:\s*none\s*!important;",
        css,
        re.DOTALL,
    )


def test_latex_homepage_fallback_is_generated(tmp_path):
    output = tmp_path / "latex"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "latex",
            "-q",
            str(DOCS),
            str(output),
        ],
        check=True,
        cwd=ROOT,
        env={**os.environ, "READTHEDOCS": "True"},
    )
    latex = read(output / "manipulapy.tex")

    for required in (
        "pip install manipulapy",
        "Learn robotics through executable notebooks",
        "robot tracing its reachable workspace",
        "NumPy",
        "Compute Backends guide",
        "Path planning",
        "AGPL compliance questions",
        "modified network services",
        "commercial licensing options",
    ):
        assert required in latex


def test_mobile_api_signatures_remain_horizontally_scrollable():
    css = read(CSS)
    mobile = css[css.index("@media (max-width: 767px)") :]

    assert re.search(
        r"\.bd-content dl\.py > dt\.sig,[^{]+\{[^}]*"
        r"max-width:\s*100%;[^}]*overflow-x:\s*auto;",
        mobile,
        re.DOTALL,
    )


def test_backend_eyebrow_is_visually_ordered_before_its_rst_heading():
    source = read(INDEX)
    css = read(CSS)

    assert re.search(
        r"^\.\. rst-class:: mp-backends\n\nCompute where your work belongs\.\n-{3,}$",
        source,
        re.MULTILINE,
    )
    assert '<section class="mp-backends"' not in source
    assert source.index("Compute where your work belongs.") < source.index(
        'class="mp-overline">One API, four array libraries'
    )
    assert re.search(
        r"\.mp-backends\s*\{[^}]*display:\s*flex;[^}]*flex-direction:\s*column;",
        css,
        re.DOTALL,
    )
    assert re.search(r"\.mp-backends\s*>\s*\.mp-overline\s*\{[^}]*order:\s*0;", css)
    assert re.search(r"\.mp-backends\s*>\s*h2\s*\{[^}]*order:\s*1;", css)
    assert re.search(r"\.mp-backends\s*>\s*\.mp-backends__grid\s*\{[^}]*order:\s*2;", css)


def test_motion_is_progressive_and_scroll_listener_free():
    script = read(MOTION)
    assert "IntersectionObserver" in script
    assert "prefers-reduced-motion: reduce" in script
    assert ".animate(" in script
    assert 'addEventListener("scroll"' not in script
    assert "anime" not in script.lower()


def test_eyebrow_budget_is_not_exceeded():
    source = read(INDEX)
    section_count = len(
        re.findall(
            r"^\.\. rst-class:: mp-(?:hero|paths|motion-gallery|backends|api-links)$",
            source,
            re.MULTILINE,
        )
    )
    eyebrow_count = source.count('class="mp-overline"')
    assert section_count == 5
    assert eyebrow_count <= 2
