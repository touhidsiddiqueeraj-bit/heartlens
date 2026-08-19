#!/usr/bin/env bash
# Build the clickable final PDF from main.tex (TeXFlow export) + references.bib.
set -e
cd "$(dirname "$0")"
grep -q '\\usepackage\[colorlinks' main.tex || \
  sed -i 's#\\usepackage{url}#\\usepackage{url}\n\\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=blue]{hyperref}#' main.tex
sed -i 's#/home/touhid/Documents/texflowmcp/workspace/fig_#figures/fig_#g' main.tex
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
echo "pages=$(pdfinfo main.pdf 2>/dev/null | grep -i pages)"
