#!/usr/bin/env bash
set -euo pipefail
xelatex -interaction=nonstopmode -halt-on-error image_processing_next_decade_ch03_fa.tex
xelatex -interaction=nonstopmode -halt-on-error image_processing_next_decade_ch03_fa.tex
if grep -Eq 'Undefined control sequence|LaTeX Error|Citation.*undefined|Reference.*undefined' image_processing_next_decade_ch03_fa.log; then
  echo 'Build completed but unresolved errors/warnings were found.' >&2
  exit 1
fi
pdfinfo image_processing_next_decade_ch03_fa.pdf | grep -E 'Pages|Page size|Title'
