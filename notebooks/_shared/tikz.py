"""TikZ/PGF figure helpers for the ManipulaPy notebook course.

Two paths, both compiled with lualatex:
  * render_tikz(code)             -> hand-authored TikZ standalone -> PNG (concept diagrams)
  * setup_pgf()/embed_pgf_fig()  -> matplotlib pgf backend        -> PNG (data plots)

PNGs are written to an output dir (default notebooks/_figures) and returned as an
IPython.display.Image for inline embedding, so they render on GitHub and survive re-runs.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

_DEF_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "_figures")

_STANDALONE = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,calc,3d,angles,quotes,decorations.markings}
\begin{document}
%s
\end{document}
"""


def _ensure(outdir):
    outdir = outdir or _DEF_OUTDIR
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _pdf_to_png(pdf_path, png_path, dpi=200):
    prefix = png_path[:-4] if png_path.endswith(".png") else png_path
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", pdf_path, prefix],
        check=True,
    )


def render_tikz(code, name, outdir=None, dpi=200):
    """Compile a TikZ body (``\\begin{tikzpicture}...\\end{tikzpicture}``) to a PNG.

    Returns an IPython.display.Image pointing at ``<outdir>/<name>.png``.
    """
    from IPython.display import Image

    outdir = _ensure(outdir)
    png = os.path.join(outdir, f"{name}.png")
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, f"{name}.tex"), "w") as f:
            f.write(_STANDALONE % code)
        subprocess.run(
            ["lualatex", "-interaction=nonstopmode", f"{name}.tex"],
            cwd=td,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _pdf_to_png(os.path.join(td, f"{name}.pdf"), png, dpi=dpi)
    return Image(filename=png)


def render_tikz_file(tex_path, name=None, outdir=None, dpi=200):
    """Compile a standalone TikZ ``.tex`` file to a PNG and return an Image.

    ``tex_path`` is a full standalone LaTeX document (it compiles on its own in any
    TeX editor). Source files live under ``notebooks/_figures/src/``; the rendered
    PNG goes to ``outdir`` (default ``notebooks/_figures``) named ``<name>.png``
    (defaults to the source file's stem).
    """
    from IPython.display import Image

    if not os.path.isabs(tex_path):
        tex_path = os.path.join(os.path.dirname(__file__), "..", tex_path)
    tex_path = os.path.abspath(tex_path)
    if not os.path.exists(tex_path):
        raise FileNotFoundError(tex_path)

    name = name or os.path.splitext(os.path.basename(tex_path))[0]
    outdir = _ensure(outdir)
    png = os.path.join(outdir, f"{name}.png")
    with open(tex_path) as f:
        doc = f.read()
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, f"{name}.tex"), "w") as f:
            f.write(doc)
        subprocess.run(
            ["lualatex", "-interaction=nonstopmode", f"{name}.tex"],
            cwd=td,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _pdf_to_png(os.path.join(td, f"{name}.pdf"), png, dpi=dpi)
    return Image(filename=png)


def setup_pgf():
    """Configure matplotlib to render via the pgf/lualatex backend; return pyplot."""
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "font.family": "serif",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "figure.figsize": (5.0, 3.2),
        }
    )
    return plt


def embed_pgf_fig(fig, name, outdir=None, dpi=200):
    """Save a matplotlib (pgf-backed) figure to PDF, convert to PNG, return Image."""
    from IPython.display import Image

    outdir = _ensure(outdir)
    pdf = os.path.join(outdir, f"{name}.pdf")
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(pdf, bbox_inches="tight")
    _pdf_to_png(pdf, png, dpi=dpi)
    return Image(filename=png)
