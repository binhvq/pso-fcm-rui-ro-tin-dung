# Dữ liệu tín dụng cho tiểu luận PSO-FCM

Thư mục này chứa dữ liệu thô dùng cho tiểu luận "Ứng dụng PSO kết hợp Fuzzy C-means trong phân cụm dữ liệu phục vụ đánh giá rủi ro tín dụng". Tất cả các bộ được tải ngày 2026-10-13 từ UCI Machine Learning Repository (không cần đăng nhập). Mỗi bộ nằm trong một thư mục riêng dưới `raw/`, gồm file zip gốc và các file đã giải nén. Không sửa file trong `raw/`; mọi bước tiền xử lý phải sinh ra file ở chỗ khác.

Giấy phép: bốn bộ trong bảng Tổng quan được UCI phát hành theo **Creative Commons Attribution 4.0 International (CC BY 4.0)** (đã kiểm tra trên trang từng bộ ngày 2026-10-13). Được dùng, chia sẻ và chỉnh sửa, với điều kiện ghi nguồn như mục "Trích dẫn".

## Tổng quan

| Thư mục | Bộ dữ liệu | UCI id | DOI | Mẫu × thuộc tính | Nhãn | Missing |
|---|---|---|---|---|---|---|
| `raw/german-credit/` | Statlog (German Credit Data) | 144 | 10.24432/C5NC77 | 1000 × 20 (7 số, 13 định danh); bản số 1000 × 24 | 1 = good 700 (70%), 2 = bad 300 (30%) | Không |
| `raw/australian-credit/` | Statlog (Australian Credit Approval) | 143 | 10.24432/C59012 | 690 × 14 (6 số, 8 định danh) | 0 = 383 (55,5%), 1 = 307 (44,5%) | Không (đã điền sẵn) |
| `raw/taiwan-default/` | Default of Credit Card Clients | 350 | 10.24432/C55S3H | 30000 × 23 (+ cột ID) | 0 = 23364 (77,9%), 1 = 6636 (22,1%) | Không |
| `raw/credit-approval-crx/` | Credit Approval (crx) | 27 | 10.24432/C5FS30 | 690 × 15 (6 số, 9 định danh) | `-` = 383 (55,5%), `+` = 307 (44,5%) | 37 dòng, 67 ô `?` |

URL gốc:

- German: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (zip: `https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip`)
- Australian: https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval (zip: `https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip`)
- Taiwan: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (zip: `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`)
- Credit Approval: https://archive.ics.uci.edu/dataset/27/credit+approval (zip: `https://archive.ics.uci.edu/static/public/27/credit+approval.zip`)

SHA-256 của file zip đã tải:

```text
e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b  raw/german-credit/statlog-german.zip
f0bd9cd9b1d133cb20456b8953e1f8139fd30ed2b3f66da540db75136f1e9e0b  raw/australian-credit/statlog-australian.zip
56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602  raw/taiwan-default/taiwan.zip
e3adfa0387815e3a9d8aaaf7b1cd7365424c83298bca6358bc48e451b4a26dd3  raw/credit-approval-crx/crx.zip
```

## 1. Statlog (German Credit Data)

File: `german.data` (20 thuộc tính gốc, phân tách bằng khoảng trắng, mã định danh dạng `A11`, `A43`...), `german.data-numeric` (24 cột số do Đại học Strathclyde mã hoá cho StatLog, cột 25 là nhãn), `german.doc` (mô tả, có ma trận chi phí).

Cột chính của `german.data`:

| Cột | Ý nghĩa | Kiểu |
|---|---|---|
| A1 | Tình trạng tài khoản vãng lai (A11 < 0 DM ... A14 không có tài khoản) | định danh, 4 mức |
| A2 | Thời hạn khoản vay (tháng), 4–72 | số |
| A3 | Lịch sử tín dụng (A30–A34) | định danh, 5 mức |
| A4 | Mục đích vay (xe, nội thất, giáo dục, kinh doanh...) | định danh, 10 mức có mặt |
| A5 | Số tiền vay (DM), 250–18424 | số, lệch phải (skew 1,95) |
| A6 | Tài khoản tiết kiệm/trái phiếu | định danh, 5 mức |
| A7 | Thời gian làm việc hiện tại | định danh có thứ tự, 5 mức |
| A8 | Tỉ lệ trả góp trên thu nhập khả dụng (1–4) | số (thứ bậc) |
| A9 | Tình trạng cá nhân và giới tính | định danh, 4 mức |
| A10 | Người đồng vay/bảo lãnh | định danh, 3 mức |
| A11 | Thời gian cư trú hiện tại (1–4) | số (thứ bậc) |
| A12 | Tài sản | định danh, 4 mức |
| A13 | Tuổi, 19–75 | số |
| A14 | Kế hoạch trả góp khác | định danh, 3 mức |
| A15 | Nhà ở (thuê/sở hữu/ở miễn phí) | định danh, 3 mức |
| A16 | Số khoản tín dụng hiện có tại ngân hàng (1–4) | số |
| A17 | Nghề nghiệp | định danh có thứ tự, 4 mức |
| A18 | Số người phụ thuộc (1–2) | số |
| A19 | Có điện thoại | nhị phân |
| A20 | Lao động nước ngoài | nhị phân (963/37) |
| class | 1 = good, 2 = bad | nhãn |

Ma trận chi phí trong `german.doc`: dự báo nhầm "bad" thành "good" có chi phí 5, nhầm ngược lại có chi phí 1.

Đã kiểm tra: 1000 dòng × 21 cột (20 thuộc tính + nhãn); 7 thuộc tính số, 13 định danh, khớp mô tả UCI. `german.data-numeric` có 1000 × 25, nhãn trùng khớp 100% với `german.data`. Không có missing, không có dòng trùng. One-hot đủ mức của 13 thuộc tính định danh cho 54 cột, cộng 7 cột số thành 61 chiều.

Lưu ý: Grömping (2019) chỉ ra bản German Credit này có lỗi mã hoá một số thuộc tính (ví dụ A9 giới tính/tình trạng hôn nhân). Bản sửa là "South German Credit" (UCI id 522, DOI 10.24432/C5X89F; bản cập nhật id 573, DOI 10.24432/C5QG88). Nhóm đã chốt dùng bản sửa làm bộ chính (mục 1b); bản này chỉ giữ để đối chiếu với các bài báo dùng bản gốc.

## 1b. South German Credit (bộ chính)

Bản sửa lỗi mã hoá của German Credit do Grömping (2019) công bố. Cùng 1.000 hồ sơ vay của một ngân hàng
Nam Đức (1973–1975), nhưng mã của các thuộc tính định danh đã được đối chiếu lại với tài liệu gốc.

- UCI id 522, DOI 10.24432/C5X89F (https://archive.ics.uci.edu/dataset/522/south+german+credit).
  Bản "UPDATE" id 573 (DOI 10.24432/C5QG88) phát cùng một file zip (đã so khớp từng byte), nên chỉ giữ một bản.
- Giấy phép: CC BY 4.0 (theo trang UCI).
- Thư mục `raw/south-german-credit/`: `SouthGermanCredit.asc` (dữ liệu, phân tách bằng khoảng trắng, có dòng
  tiêu đề), `codetable.txt` (bảng mã từng giá trị), `read_SouthGermanCredit.R` (script đọc của tác giả,
  không chạy; chỉ đọc để tham khảo tên cột).
- SHA-256 zip: `0b40d40eb7321693d559e247a556f88a6cc8df8489c3cb2ae084db7592584551  raw/south-german-credit/south-german-credit.zip`

Đã kiểm tra: 1000 × 21, không có missing; nhãn `kredit`: 1 = tốt 700 (70%), 0 = xấu 300 (30%). Lưu ý
mã nhãn khác bản gốc (bản gốc 1 = tốt, 2 = xấu).

Tên cột tiếng Đức → tiếng Anh (theo `codetable.txt`) và kiểu:

| Cột | Tên tiếng Anh | Kiểu | Số mức |
|---|---|---|---|
| laufkont | status (tài khoản vãng lai) | thứ bậc | 4 |
| laufzeit | duration (tháng) | số | 33 |
| moral | credit_history | định danh | 5 |
| verw | purpose | định danh | 10 |
| hoehe | amount (DM) | số | 923 |
| sparkont | savings | thứ bậc | 5 |
| beszeit | employment_duration | thứ bậc | 5 |
| rate | installment_rate | thứ bậc | 4 |
| famges | personal_status_sex (nhạy cảm) | định danh | 4 |
| buerge | other_debtors | định danh | 3 |
| wohnzeit | present_residence | thứ bậc | 4 |
| verm | property | thứ bậc | 4 |
| alter | age | số | 53 |
| weitkred | other_installment_plans | định danh | 3 |
| wohn | housing | định danh | 3 |
| bishkred | number_credits | thứ bậc | 4 |
| beruf | job | thứ bậc | 4 |
| pers | people_liable | nhị phân | 2 |
| telef | telephone | nhị phân | 2 |
| gastarb | foreign_worker (nhạy cảm) | nhị phân | 2 |
| kredit | credit_risk (nhãn, chỉ dùng để diễn giải cụm) | nhị phân | 2 |

Gợi ý tiền xử lý: one-hot cho cột định danh, giữ thứ bậc dưới dạng số, log1p cho `hoehe`, z-score toàn bộ;
không đưa `kredit` vào phân cụm. Cân nhắc loại `famges`, `gastarb` khỏi đặc trưng phân cụm vì là thuộc
tính nhạy cảm.

Trích dẫn: Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Reports in
Mathematics, Physics and Chemistry, Report 4/2019, Beuth University of Applied Sciences Berlin.

## 2. Statlog (Australian Credit Approval)

File: `australian.dat` (14 thuộc tính + nhãn, phân tách bằng khoảng trắng), `australian.doc`.

Tên và giá trị thuộc tính đã bị ẩn danh hoá (A1–A14) để bảo mật, nên không diễn giải được nghĩa nghiệp vụ. Thuộc tính số: A2, A3, A7, A10, A13, A14. Thuộc tính định danh: A1, A4, A5, A6, A8, A9, A11, A12 (A5 có 14 mức, A6 có 8 mức). Nhãn trong file là 0/1 (tài liệu ghi 1/2): 0 = 383, 1 = 307, khớp phân bố 55,5% / 44,5% trong tài liệu.

Đã kiểm tra: 690 × 15, khớp UCI. Không có missing: theo `australian.doc`, 37 dòng thiếu đã được điền bằng mode (định danh) và mean (số). Thuộc tính số rất lệch, đặc biệt A14 (max 100001, skew 13,1) và A10 (skew 5,2).

Đây là cùng dữ liệu gốc với Credit Approval (crx) sau khi mã hoá lại: thống kê các cột số trùng nhau (cùng mean, min, max, skew).

## 3. Default of Credit Card Clients (Taiwan)

File: `default of credit card clients.xls` (Excel 97–2003). Hàng 1 là mã X1…X23/Y, hàng 2 là tên cột, nên đọc bằng `pandas.read_excel(..., header=1)`; cần gói `xlrd`.

| Cột | Ý nghĩa |
|---|---|
| ID | Mã khách hàng (duy nhất, không dùng để phân cụm) |
| LIMIT_BAL | Hạn mức tín dụng (NT$), 10000–1000000 |
| SEX | 1 = nam, 2 = nữ |
| EDUCATION | 1 = cao học, 2 = đại học, 3 = trung học, 4 = khác; có các giá trị 0, 5, 6 không được mô tả (345 dòng) |
| MARRIAGE | 1 = kết hôn, 2 = độc thân, 3 = khác; giá trị 0 không được mô tả (54 dòng) |
| AGE | Tuổi, 21–79 |
| PAY_0, PAY_2…PAY_6 | Tình trạng trả nợ từ tháng 9 lùi về tháng 4/2005 (-2, -1, 0 = trả đúng hạn hoặc không phát sinh; 1–8 = trễ 1–8 tháng). Tên PAY_0 thực chất là tháng thứ nhất và không có cột PAY_1 |
| BILL_AMT1…6 | Dư nợ sao kê tháng 9 → tháng 4 (NT$); 1930 dòng có giá trị âm |
| PAY_AMT1…6 | Số tiền đã trả tháng 9 → tháng 4 (NT$); rất lệch (skew 10–30) |
| default payment next month | 1 = vỡ nợ tháng kế tiếp, 0 = không |

Đã kiểm tra: 30000 × 25 (ID + 23 thuộc tính + nhãn), khớp UCI. Không có missing, ID duy nhất; có 35 dòng trùng nhau nếu bỏ ID. Tỉ lệ vỡ nợ 22,12%. Mẫu con phân tầng theo nhãn (random_state=42) giữ đúng tỉ lệ: n = 2000 → 22,10%, n = 3000 → 22,13%, n = 5000 → 22,12%.

## 4. Credit Approval (crx)

File: `crx.data` (CSV không header, missing ký hiệu `?`), `crx.names`, `credit.names`, `credit.lisp` (bộ Japanese Credit Screening cũ đi kèm zip, không dùng).

Thuộc tính số: A2, A3, A8, A11, A14, A15. Thuộc tính định danh: A1, A4, A5, A6, A7, A9, A10, A12, A13. Nhãn A16: `+` = 307, `-` = 383.

Đã kiểm tra: 690 × 16, khớp UCI. Có 37 dòng chứa missing, tổng 67 ô: A1 12, A2 12, A4 6, A5 6, A6 9, A7 9, A14 13, khớp `crx.names`. Ý nghĩa thuộc tính cũng bị ẩn danh như bản Australian.

## Gợi ý tiền xử lý cho phân cụm

- Không đưa nhãn (class/default) vào ma trận đặc trưng. Nhãn chỉ dùng sau khi phân cụm để tính ARI, NMI, purity và tỉ lệ vỡ nợ theo cụm.
- German: dùng `german.data-numeric` (24 cột số) để có cấu hình đơn giản, hoặc dùng `german.data` với one-hot cho định danh + chuẩn hoá thuộc tính số (61 chiều). Nên log1p A5 (số tiền vay) trước khi chuẩn hoá.
- Australian/crx: nên chọn một trong hai, không dùng cả hai vì là cùng dữ liệu. Nếu dùng crx thì phải tự điền missing (mode/median). Nên log1p A14/A15 và A10/A11 trước khi chuẩn hoá.
- Taiwan: dùng 14 cột số (LIMIT_BAL, AGE, BILL_AMT1–6, PAY_AMT1–6) cộng PAY_0…PAY_6 như biến thứ bậc. SEX/EDUCATION/MARRIAGE có thể bỏ hoặc one-hot (gộp EDUCATION 0/5/6 vào 4, MARRIAGE 0 vào 3). Dùng log1p cho PAY_AMT; với BILL_AMT có giá trị âm thì dùng `sign(x)·log1p(|x|)`. Bỏ ID.
- Chuẩn hoá: z-score sau khi biến đổi log là lựa chọn mặc định cho FCM/K-means (khoảng cách Euclid). Min-max rất nhạy với ngoại lai như A14 = 100001 hoặc PAY_AMT2 = 1,68 triệu, nên chỉ dùng sau khi đã log hoặc cắt ngoại lai. Phải dùng cùng một phép chuẩn hoá cho cả ba thuật toán.
- Taiwan và chi phí PSO-FCM: mỗi lần đánh giá fitness tốn O(n·c·d), khoảng 30000 × 3 × 23 ≈ 2·10^6 phép tính. Nếu viết bằng numpy vector hoá thì chạy cả 30000 mẫu vẫn khả thi. Tuy vậy, thí nghiệm cần nhiều seed và nhiều giá trị c, còn các chỉ số kiểu silhouette tốn O(n²). Vì vậy nên dùng mẫu con phân tầng 5000 (random_state cố định) cho thí nghiệm chính, rồi chạy kiểm chứng một lần trên toàn bộ 30000 mẫu nếu còn thời gian.

## Trích dẫn

- Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- Quinlan, R. (1987). *Statlog (Australian Credit Approval)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59012
- Yeh, I-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
  - Bài báo giới thiệu: Yeh, I-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- Quinlan, J. R. (1987). *Credit Approval* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5FS30
- (Nếu dùng bản sửa của German) Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Reports in Mathematics, Physics and Chemistry 4/2019, Beuth University of Applied Sciences Berlin. Dataset: https://doi.org/10.24432/C5X89F

## Dữ liệu Việt Nam

Khảo sát ngày 2026-10-13. Báo cáo khảo sát đầy đủ (kể cả những bộ không tải) là tài liệu làm việc nội bộ của nhóm, không nộp kèm.

**Không có bộ tín dụng cá nhân Việt Nam nào vừa công khai vừa có giấy phép rõ ràng.** Bộ duy nhất gần đúng là dữ liệu Kalapa Credit Scoring Challenge (2020), nhưng dữ liệu gốc chỉ được phát cho người đăng ký cuộc thi. Các bản sao trên GitHub và Kaggle do người dự thi tự đăng lại, không có giấy phép (Kaggle ghi "Unknown"). Vì vậy không tải bộ này.

### 5. Financial distress của doanh nghiệp niêm yết Việt Nam (rủi ro tín dụng doanh nghiệp)

Thư mục: `raw/vn-financial-distress-listed-firms/`. File: `data_XAI_12102022.csv` (CSV, mã hoá cp1258/latin-1), `SOURCE_COMMIT.txt` (commit nguồn `a7eb6070283feb9f4a8b02e7c4981d3849f72c4e`).

- Nguồn: https://github.com/anhle32/Explainable-Machine-Learning- . Đây là kho do nhóm tác giả công bố trong mục Data Availability Statement của bài báo Tran và cộng sự (2022), tạp chí *Data* (MDPI), doi:10.3390/data7110160.
- Giấy phép: bài báo phát hành theo CC BY 4.0, nhưng kho GitHub không ghi giấy phép riêng. Tác giả chủ động công bố dữ liệu để người khác tái lập, và số liệu gốc là chỉ số tài chính từ báo cáo tài chính công khai của công ty niêm yết. Dùng cho học thuật kèm trích dẫn bài báo là hợp lý. Không phân phối lại file.
- SHA-256: `128cd00f32cdbdfe14634452ed554dcc7e3740e808e79b266f77dbb82b06c9b4`.
- Theo bài báo: lấy từ báo cáo tài chính của công ty trên HOSE, HNX và UPCOM, loại trừ tài chính, bảo hiểm và ngân hàng. Dữ liệu "đã xử lý missing và ngoại lai". Nhãn 1 (khó khăn tài chính) được gán khi công ty có vốn chủ sở hữu âm, hoặc EBITDA/lãi vay < 1 trong hai năm liên tiếp, hoặc lợi nhuận hoạt động âm ba năm liên tiếp; nhóm tác giả có đối chiếu thêm ý kiến kiểm toán về khả năng hoạt động liên tục.

Đã kiểm tra (script `python3 -I`):

- Kích thước 3277 × 31: 6 cột định danh/nhãn và 25 chỉ số X1–X25. Không có ô trống, không có giá trị phi số.
- Nhãn `Financial_Distress`: 0 = 2841 (86,7%), 1 = 436 (13,3%), khớp bài báo.
- Gồm 825 công ty. Sàn: UPCOM 1428, HNX 987, HOSE 862 dòng. Năm 2010–2020 (bài báo ghi 2010–2021 nhưng file không có 2021); năm 2015 chỉ có 16 dòng.
- Nhãn gán theo công ty, không theo từng năm: không công ty nào có nhãn đổi qua các năm (95 công ty mang nhãn 1). Vì vậy nhãn có nghĩa là "công ty thuộc nhóm gặp khó khăn tài chính", không phải "gặp khó khăn tài chính trong năm đó".
- Cột `id` không duy nhất (128 giá trị lặp). Khoá đúng là (`Company`, `year`): không có dòng trùng.
- Cột `Name` bị mất dấu tiếng Việt ngay trong file gốc (ví dụ `G? Tr??ng Thành`). Dùng `Company` (mã chứng khoán) để tra cứu.
- Các cột đã bị cắt (winsorize) ở phân vị 1% và 99%: giá trị min bằng p1 và max bằng p99 ở mọi cột.
- **Số 0 thay cho missing.** X6 = 0 trong khi X8 > 0 ở 1084 dòng, điều này vô lý vì tổng nợ/VCSH luôn ≥ nợ ngắn hạn/VCSH. Tương tự X7 = 0 trong khi X9 > 0 ở 1074 dòng. X12 (tổng tài sản/tổng nợ) = 0 ở 1287 dòng, cũng không thể xảy ra. Có 1996/3277 dòng có ít nhất một số 0 kiểu này ở X6, X7 hoặc X12. Riêng X4 = 0 (2038 dòng) và X5 = 0 (2094 dòng) có thể là thật (công ty không có nợ dài hạn).

Ý nghĩa cột (Phụ lục A của bài báo):

| Cột | Ý nghĩa | Nhóm |
|---|---|---|
| id, year, Company, Name, Exchange | Số thứ tự, năm, mã CK, tên công ty, sàn | Định danh (không đưa vào phân cụm) |
| Financial_Distress | 1 = khó khăn tài chính, 0 = bình thường | Nhãn |
| X1 / X2 / X3 | Tỉ số tiền mặt / thanh toán nhanh / thanh toán hiện hành | Rủi ro thanh khoản |
| X4 / X5 | Nợ dài hạn / VCSH; nợ dài hạn / tổng tài sản | Rủi ro tài chính |
| X6 / X7 | Tổng nợ / VCSH; tổng nợ / tổng tài sản | Rủi ro tài chính |
| X8 / X9 | Nợ ngắn hạn / VCSH; nợ ngắn hạn / tổng tài sản | Rủi ro tài chính |
| X10 / X11 | Phải trả người bán / VCSH; phải trả người bán / tổng tài sản | Rủi ro kinh doanh |
| X12 | Tổng tài sản / tổng nợ | Rủi ro kinh doanh |
| X13 | EBITDA / (nợ ngắn hạn + lãi vay) | Rủi ro kinh doanh |
| X14 / X15 | P/E; P/E pha loãng | Thị trường |
| X16 / X17 / X18 | P/B; P/S; giá / giá trị sổ sách hữu hình | Thị trường |
| X19 | Vốn hoá thị trường (đơn vị không ghi; trung vị 170,5, có lẽ là tỉ đồng) | Thị trường |
| X20 | Giá / dòng tiền (P/CF) | Thị trường |
| X21 / X22 / X23 / X24 | EV; EV/doanh thu; EV/EBITDA; EV/EBIT | Định giá |
| X25 | EPS pha loãng (đồng) | Định giá |

Thuộc tính nhạy cảm: không có dữ liệu cá nhân. Tên và mã công ty là thông tin công khai, nhưng gắn nhãn "khó khăn tài chính" cho công ty có tên thật là thông tin dễ gây hiểu lầm. Khi trình bày, chỉ nêu tên công ty nếu thực sự cần.

Gợi ý tiền xử lý cho phân cụm:

- Ma trận đặc trưng chỉ lấy X1–X25. Bỏ id, year, Company, Name, Exchange và nhãn.
- Xử lý số 0 giả ở X6, X7, X12: coi là missing rồi điền trung vị, hoặc bỏ hẳn ba cột này vì đã có X8–X11 mang thông tin đòn bẩy tương tự. Nên ghi rõ lựa chọn trong Chương 3. Nếu giữ nguyên số 0, cụm có thể tách theo "có số 0 hay không" thay vì theo rủi ro.
- Bỏ cột gần trùng (|r| > 0,94): X15 (r = 0,999 với X14), X18 (0,96 với X16), X21 (0,95 với X19), X3 (0,944 với X2). X8 và X10 cũng có r = 0,966.
- log1p cho các cột lệch phải và không âm: X13, X19, X21. Với X20, X23, X24, X25 (có giá trị âm) dùng `sign(x)·log1p(|x|)`. Sau đó z-score cho cả ba thuật toán.
- Dữ liệu là bảng panel (công ty × năm). Có hai cách: phân cụm toàn bộ firm-year rồi tính tỉ lệ nhãn 1 theo cụm, hoặc chỉ lấy một năm (ví dụ 2018, 494 dòng, 39 nhãn 1) để tránh một công ty xuất hiện nhiều lần. Kích thước 3277 × khoảng 18–25 chiều đủ nhỏ để PSO-FCM chạy nhiều seed.
- Khi diễn giải, phải nói rõ đây là rủi ro tín dụng doanh nghiệp (khả năng trả nợ của công ty), khác các bộ UCI là tín dụng cá nhân.

Trích dẫn:

- Tran, K. L., Le, H. A., Nguyen, T. H., & Nguyen, D. T. (2022). Explainable Machine Learning for Financial Distress Prediction: Evidence from Vietnam. *Data, 7*(11), 160. https://doi.org/10.3390/data7110160 . Dữ liệu: https://github.com/anhle32/Explainable-Machine-Learning- (commit a7eb607, truy cập 2026-10-13).

## Bộ chẩn đoán bổ sung (tải 13/10/2026)

| Thư mục | Nguồn | Giấy phép | Dùng |
|---|---|---|---|
| `raw/polish-bankruptcy/` (1year–5year.arff) | Tomczak, S. (2016). Polish Companies Bankruptcy. UCI ML Repository, id 365, https://doi.org/10.24432/C5F600 | CC BY 4.0 | Chẩn đoán số cực tiểu (P15); dùng file `3year.arff`, mẫu phân tầng 5.000 |
| `raw/taiwan-bankruptcy/data.csv` | Taiwanese Bankruptcy Prediction (2020). UCI ML Repository, id 572, https://doi.org/10.24432/C5004D | CC BY 4.0 | Chẩn đoán số cực tiểu (P15); FCM m = 2 sụp |

SHA-256 của file zip hai bộ chẩn đoán bổ sung:

```text
17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9  raw/polish-bankruptcy/polish.zip
c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac  raw/taiwan-bankruptcy/tw.zip
```

## Bộ bổ sung tải ngày 2026-10-14 (chỉ dùng để chẩn đoán số cực tiểu, `thu-nghiem-so-bo/chan_doan_bo_moi.py`)

Nguồn và giấy phép do nhóm tra cứu ngày 14/10/2026; giấy phép trên OpenML do người tải lên tự khai. Các bộ trong bảng này không đưa lên kho mã công khai vì giấy phép chưa rõ hoặc không dùng trong kết quả chính; tải lại theo URL bên dưới.

| Thư mục | Bộ dữ liệu | URL tải | Giấy phép | Nhãn dương |
|---|---|---|---|---|
| `raw/give-me-some-credit/` | Give Me Some Credit (Kaggle 2011, bản OpenML) | https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff | OpenML ghi "Public"; điều lệ Kaggle chưa kiểm | FinancialDistressNextTwoYears = Yes |
| `raw/hmeq/` | HMEQ (Baesens và cộng sự 2016, DOI 10.1002/9781119449560) | https://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv | Không ghi; xin trích sách | BAD = 1 |
| `raw/american-bankruptcy/` | American Companies Bankruptcy (Lombardo và cộng sự 2022, DOI 10.3390/fi14080244) | https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv | CC BY 4.0 | status_label = failed |
| `raw/fico-heloc/` | FICO HELOC (Explainable ML Challenge 2018, bản OpenML) | https://openml.org/data/v1/download/22116521/FICO-HELOC-cleaned.arff | Không rõ | RiskPerformance = Bad |
| `raw/lending-club-openml/` | Lending Club 2007–2010 bản nhỏ (OpenML 43729) | https://openml.org/data/v1/download/22102554/Lending-Club-Loan-Data.arff | OpenML ghi "Public" | not.fully.paid = 1 |

SHA-256:

```text
92c897782a049feef5de124c1e5b82a61ed9373aa7e78565f03aea3aa3be21da  give-me-some-credit/GiveMeSomeCredit.arff
dfdbc2b7cdf728a15b53e323cde6127995715dfa6b178bd3c1e3d9916d0367aa  hmeq/hmeq.csv
d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a  american-bankruptcy/american_bankruptcy_dataset.csv
a9472491bc142aab47a26aa16d51227aeb942e7ebf3574b08de8ae51accacf39  fico-heloc/FICO-HELOC-cleaned.arff
a48a56b1aedb9869f7a1541c8b58d7ab3df7599c5674faa17dffbb088c0a2821  lending-club-openml/Lending-Club-Loan-Data.arff
```
