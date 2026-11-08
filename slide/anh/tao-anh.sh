#!/usr/bin/env bash
# Chuyển các hình trong hinh/ mà slide dùng sang PNG 220 dpi (cần pdftoppm của poppler).
set -euo pipefail
cd "$(dirname "$0")"

for ten in e7-quy-dao-gbest e8-hoi-tu e8-trong-so e9-quy-dao; do
  pdftoppm -r 220 -png -singlefile "../../hinh/$ten.pdf" "$ten"
done
echo "Đã ghi ảnh vào slide/anh/"
