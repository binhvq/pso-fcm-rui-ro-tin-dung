# Ứng dụng PSO kết hợp Fuzzy C-means trong phân cụm dữ liệu phục vụ đánh giá rủi ro tín dụng

Tiểu luận môn Khai phá dữ liệu và phát hiện tri thức (IT7314001), Cao học HTTT K16.1, Trường Đại học Công nghiệp Hà Nội. Nhóm 1:

- Vương Quốc Bình
- Trần Trường Giang
- Nguyễn Quang Hà
- Nguyễn Tiến Hải

## Nội dung

| Đường dẫn | Nội dung |
|---|---|
| `build/main.pdf` | Bản PDF tiểu luận |
| `main.tex`, `noi-dung/` | Nguồn LaTeX (bìa, các chương, phụ lục, tài liệu tham khảo) |
| `results/` | Kết quả thực nghiệm E1–E10 (CSV), bảng LaTeX `bang/*.tex` và số liệu `so-lieu-*.tex` do code sinh ra |
| `hinh/` | Hình trong tiểu luận |
| `code/` | Mã nguồn Python: FCM, các biến thể PSO, PSO tìm tâm và tìm trọng số, chỉ số đánh giá, trình chạy thực nghiệm, phép thử |
| `thu-nghiem-so-bo/` | Thí nghiệm thử trước khi chốt hướng và kết quả của chúng |
| `slide/` | Slide báo cáo (3 phần và bản gộp `.pptx`, bản xem PDF) cùng script sinh slide |
| `data/` | Dữ liệu thô dùng trong thực nghiệm và mô tả nguồn, giấy phép (`data/README.md`) |

## Chạy thực nghiệm

```bash
pip install -r requirements.txt pytest
cd code
python3 chay_thuc_nghiem.py --thi-nghiem tat-ca   # E1–E4; E5–E10: chay_e5.py ... chay_e10.py
python3 -m pytest -q kiem_thu_*.py
```

Dữ liệu doanh nghiệp Việt Nam (Tran và cộng sự, 2022) không được phân phối lại nên không có trong kho. Tải `data_XAI_12102022.csv` từ https://github.com/anhle32/Explainable-Machine-Learning- (commit `a7eb607`) và đặt vào `data/raw/vn-financial-distress-listed-firms/`. Phép thử hồi quy tiền xử lý cần thêm tệp mẫu `code/mau_kiem_thu/hoi-quy-X.npz`, tệp này chứa dữ liệu Việt Nam đã chuẩn hoá nên cũng không đưa lên.

## Biên dịch tiểu luận

Chép các font Times New Roman (`times*.ttf`) và Courier New (`cour*.ttf`) vào thư mục `fonts/`, rồi chạy `./compile.sh` (cần Docker hoặc `xelatex` + `biber`). Kết quả ở `build/main.pdf`.

## Sinh lại slide

```bash
cd slide
npm ci
npm run slide
```

Lệnh này chuyển hình `hinh/*.pdf` sang `anh/*.png`, biên dịch công thức `ct/ct.tex` thành `ct/ct-*.png`, xuất số liệu từ `results/so-lieu*.tex` ra `so-lieu.json`, rồi dựng các tệp `.pptx`. Cần Node.js, `xelatex`, `pdftoppm` (poppler) và ImageMagick. Slide dùng font Be Vietnam Pro.
