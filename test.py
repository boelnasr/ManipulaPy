#!/usr/bin/env python3
from subprocess import run

# LaTeX code to a .tex file
latex_code = r"""
\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{positioning, shapes, fit, arrows.meta}

\begin{document}
\begin{figure*}[ht]
\centering

\resizebox{\textwidth}{!}{
\begin{tikzpicture}[
  box/.style={
    rectangle,
    draw,
    minimum width=2.2cm,
    minimum height=1.2cm,
    rounded corners,
    thick,
    align=center
  },
  widebox/.style={
    rectangle,
    draw,
    minimum width=3cm,
    minimum height=1.2cm,
    rounded corners,
    fill=blue!10,
    thick,
    align=center
  },
  envbox/.style={
    rectangle,
    draw,
    minimum width=2.5cm,
    minimum height=1.2cm,
    rounded corners,
    fill=orange!30,
    thick,
    align=center
  },
  replaybox/.style={
    rectangle,
    draw,
    minimum width=2.5cm,
    minimum height=1.2cm,
    rounded corners,
    fill=yellow!30,
    thick,
    align=center
  },
  herbox/.style={
    rectangle,
    draw,
    minimum width=2.5cm,
    minimum height=1.2cm,
    rounded corners,
    fill=green!20,
    thick,
    align=center
  },
  pdbox/.style={
    regular polygon,
    regular polygon sides=6,
    draw,
    fill=purple!10,
    thick,
    minimum size=1.5cm,
    align=center
  },
  arrow/.style={
    -Stealth,
    thick
  },
  doublearrow/.style={
    <->,
    thick,
    dashed
  },
  node distance=3cm and 4cm,
  every node/.append style={font=\small}
]

% NODES
\node[envbox] (environment) { Environment\\ $\mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}'$ };
\node[herbox, below=2cm of environment] (her) { HER Filter\\ Goal Relabeling };
\node[replaybox, below=2cm of her] (replay) { Prioritized\\ Replay Buffer };
\node[box, right=8.0cm of environment, yshift=3.0cm] (actor1) { Actor 1\\ $\pi_{\theta_1}(a_1|s_1)$ };
\node[box, right=8.0cm of environment] (actor2) { Actor 2\\ $\pi_{\theta_2}(a_2|s_2)$ };
\node[box, right=8.0cm of environment, yshift=-3.0cm] (actorN) { Actor n\\ $\pi_{\theta_n}(a_n|s_n)$ };
\node[box, below=3.0cm of environment] (critic) { Centralized Critic\\ $V_\phi(s)$ };

\end{tikzpicture}
} 

\caption{Enhanced MAPPO Architecture}
\end{figure*}
\end{document}
"""

# File paths
output_dir = "/home/abo/Desktop"
tex_file_path = f"{output_dir}/MAPPO_diagram.tex"
pdf_file_path = f"{output_dir}/MAPPO_diagram.pdf"
svg_file_path = f"{output_dir}/MAPPO_diagram.svg"

# Save LaTeX code to a file
with open(tex_file_path, "w") as f:
    f.write(latex_code)

# Compile the LaTeX file to PDF
run(["pdflatex", "-interaction=nonstopmode", "-output-directory", output_dir, tex_file_path], check=True)

# Convert PDF to SVG using pdf2svg
run(["pdf2svg", pdf_file_path, svg_file_path], check=True)

print(f"SVG file generated at: {svg_file_path}")
