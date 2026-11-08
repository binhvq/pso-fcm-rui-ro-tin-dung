#!/usr/bin/env bash
# ============================================================
# compile.sh — Biên dịch tiểu luận
# Ưu tiên Docker TeX Live (image texlive/texlive, cùng bộ Overleaf dùng). Nếu
# Docker không chạy thì dùng xelatex + biber cài sẵn trên máy.
#
# Cách dùng:
#   ./compile.sh         # biên dịch ra build/main.pdf
#   ./compile.sh clean   # xoá toàn bộ thư mục build/
# ============================================================
set -euo pipefail

# Chạy từ thư mục chứa script này (thư mục gốc project) để Path=fonts/ đúng.
cd "$(dirname "$0")"

IMAGE="texlive/texlive:latest"

if [[ "${1:-}" == "clean" ]]; then
  rm -rf build/
  echo "Đã xoá build/"
  exit 0
fi

mkdir -p build

if docker info >/dev/null 2>&1; then
  # --user + HOME=/tmp: file build/ thuộc về user host (không phải root),
  # và xetex có thư mục ghi cache font.
  # latexmk tự lo vòng xelatex -> biber -> xelatex -> xelatex.
  docker run --rm \
    -v "$PWD":/work -w /work \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    "$IMAGE" \
    latexmk -xelatex -outdir=build -interaction=nonstopmode main.tex
elif command -v xelatex >/dev/null && command -v biber >/dev/null; then
  echo "Docker không chạy → dùng xelatex + biber trên máy" >&2
  run_xelatex() { xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >/dev/null; }
  run_xelatex
  biber --quiet --input-directory=build --output-directory=build main
  run_xelatex
  run_xelatex
else
  echo "Cần Docker đang chạy hoặc xelatex + biber trên máy" >&2
  exit 1
fi

echo "Xong: build/main.pdf"
