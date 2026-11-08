#!/usr/bin/env bash
# Biên dịch ct.tex (8 công thức, mỗi trang một công thức) và cắt thành ct-1.png ... ct-8.png cho tao-slide.js.
# Cần xelatex (font TeX Gyre Termes), pdftoppm (poppler) và ImageMagick.
set -euo pipefail
cd "$(dirname "$0")"

xelatex -interaction=nonstopmode -halt-on-error ct.tex >/dev/null
tam=$(mktemp -d)
trap 'rm -rf "$tam"' EXIT
pdftoppm -r 300 -png ct.pdf "$tam/trang"
i=1
for trang in $(ls "$tam"/trang-*.png | sort -V); do
  magick "$trang" -trim +repage -bordercolor white -border 20 "ct-$i.png"
  i=$((i + 1))
done
echo "Đã ghi $((i - 1)) ảnh ct-*.png"
