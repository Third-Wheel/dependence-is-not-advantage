#!/bin/sh
# Build Figure 1 for the project page, in a light and a dark version: an SVG of each
# for the page, and a PNG of the light one for link previews.
set -e
cd "$(dirname "$0")"
pdflatex -halt-on-error -interaction=nonstopmode figure-1.tex > /dev/null
pdflatex -halt-on-error -interaction=nonstopmode -jobname=figure-1-dark \
  '\def\darkmode{}\input{figure-1.tex}' > /dev/null
pdftocairo -svg figure-1.pdf ../site/figure-1.svg
pdftocairo -svg figure-1-dark.pdf ../site/figure-1-dark.svg
pdftocairo -png -r 200 -singlefile figure-1.pdf ../site/figure-1
rm -f figure-1*.aux figure-1*.log figure-1*.pdf
