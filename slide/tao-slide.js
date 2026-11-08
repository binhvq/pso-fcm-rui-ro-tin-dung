// Sinh slide thuyết trình tiểu luận PSO–FCM, chia theo người trình bày.
//   npm run slide              -> phan-*.pptx và tong-hop-nhom-1-pso-fcm.pptx (sinh lại ảnh, công thức, số liệu trước)
//   node tao-slide.js          -> chỉ dựng slide từ anh/, ct/ và so-lieu.json có sẵn
// Số liệu lấy từ so-lieu.json (node xuat-so-lieu.js xuất từ results/so-lieu*.tex); khóa {tênMacro} không tồn tại thì dừng.
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

const GOC = __dirname;
const SL = JSON.parse(fs.readFileSync(path.join(GOC, "so-lieu.json"), "utf8"));
const s = (t) => t.replace(/\{(soLieu\w+)\}/g, (_, k) => {
  if (!(k in SL)) throw new Error(`Thiếu macro ${k}`);
  return SL[k].replace(/\\_/g, "_").replace(/\\%/g, "%");
});
const n = (k) => s(`{${k}}`);

const C = { navy: "0B2A5B", tint: "EAF0F8", amber: "D9822B", ink: "1F2937", muted: "5B6475", white: "FFFFFF",
  line: "C9D3E3", pale: "F6F8FC", green: "2F7D5B" };
const HF = "Be Vietnam Pro", BF = "Be Vietnam Pro";
const FONT_BD = { titleFontFace: BF, legendFontFace: BF, catAxisLabelFontFace: BF, valAxisLabelFontFace: BF, dataLabelFontFace: BF };
const W = 13.333, H = 7.5;

function kichThuocPng(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
const ANH = (t) => path.join(GOC, "anh", t);
const CT = (i) => path.join(GOC, "ct", `ct-${i}.png`);

function anh(slide, p, x, y, { w, h }) {
  const k = kichThuocPng(p);
  if (w && !h) h = (w * k.h) / k.w;
  else if (h && !w) w = (h * k.w) / k.h;
  else { const r = Math.min(w / k.w, h / k.h); const w2 = k.w * r, h2 = k.h * r; x += (w - w2) / 2; y += (h - h2) / 2; w = w2; h = h2; }
  slide.addImage({ path: p, x, y, w, h, altText: path.basename(p) });
  return { x, y, w, h };
}

const PHAN = {
  1: { nguoi: "Nguyễn Quang Hà", nhan: "Phần 1 · Lý thuyết" },
  2: { nguoi: "Nguyễn Tiến Hải", nhan: "Phần 2 · Thuật toán, mô hình" },
  3: { nguoi: "Trần Trường Giang", nhan: "Phần 3 · Thực nghiệm, kết luận" },
};

function trang(pres, phan, tieuDe) {
  const sl = pres.addSlide();
  sl.background = { color: C.white };
  sl.addText(tieuDe, { x: 0.6, y: 0.35, w: 9.4, h: 0.9, fontFace: HF, fontSize: 28, bold: true, color: C.navy,
    valign: "middle", margin: 0, fit: "shrink" });
  const p = PHAN[phan];
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.25, y: 0.45, w: 2.5, h: 0.7, fill: { color: C.tint },
    line: { color: C.tint }, rectRadius: 0.12 });
  sl.addText([{ text: p.nhan, options: { fontSize: 10.5, color: C.muted, breakLine: true } },
    { text: p.nguoi, options: { fontSize: 12.5, bold: true, color: C.navy } }],
    { x: 10.35, y: 0.47, w: 2.3, h: 0.66, fontFace: BF, align: "center", valign: "middle", margin: 0 });
  sl.addText("Nhóm 1 · Ứng dụng PSO kết hợp Fuzzy C-means trong đánh giá rủi ro tín dụng",
    { x: 0.6, y: 7.0, w: 9, h: 0.3, fontFace: BF, fontSize: 9.5, color: C.muted, margin: 0 });
  sl.slideNumber = { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontFace: BF, fontSize: 9.5, color: C.muted, align: "right" };
  return sl;
}

function soTron(pres, sl, x, y, d, so, mau = C.navy) {
  sl.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: mau }, line: { color: mau } });
  sl.addText(String(so), { x, y, w: d, h: d, fontFace: HF, fontSize: d * 26, bold: true, color: C.white,
    align: "center", valign: "middle", margin: 0 });
}

function the(pres, sl, x, y, w, h, tieu, than, { mau = C.tint, so = null, coTieu = 17, coThan = 16 } = {}) {
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: mau }, line: { color: mau }, rectRadius: 0.1 });
  let lx = x + 0.25;
  if (so !== null) { soTron(pres, sl, x + 0.22, y + 0.22, 0.5, so); lx = x + 0.9; }
  sl.addText(s(tieu), { x: lx, y: y + 0.18, w: w - (lx - x) - 0.2, h: 0.55, fontFace: HF, fontSize: coTieu, bold: true,
    color: C.navy, valign: "middle", margin: 0 });
  const runs = (Array.isArray(than) ? than : [than]).map((t, i, a) => ({ text: s(t),
    options: { bullet: Array.isArray(than) && a.length > 1 ? { indent: 14 } : false, breakLine: i < a.length - 1,
      paraSpaceAfter: 4 } }));
  sl.addText(runs, { x: x + 0.25, y: y + 0.8, w: w - 0.45, h: h - 0.95, fontFace: BF, fontSize: coThan, color: C.ink,
    valign: "top", margin: 0 });
}

function soLon(sl, x, y, w, so, nhan, mau = C.navy, co = 40) {
  sl.addText(s(so), { x, y, w, h: 0.95, fontFace: HF, fontSize: co, bold: true, color: mau, margin: 0, valign: "bottom" });
  sl.addText(s(nhan), { x, y: y + 1.0, w, h: 0.8, fontFace: BF, fontSize: 13, color: C.muted, margin: 0, valign: "top" });
}

function gach(sl, items, box, co = 16) {
  sl.addText(items.map((t, i) => ({ text: s(t), options: { bullet: { indent: 16 }, breakLine: i < items.length - 1,
    paraSpaceAfter: 8 } })), { ...box, fontFace: BF, fontSize: co, color: C.ink, valign: "top", margin: 0 });
}

function muiTen(pres, sl, x, y, w) {
  sl.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.42, fill: { color: C.amber }, line: { color: C.amber } });
}

function ghiChu(sl, t) { sl.addNotes(s(t).replace(/\n\s+/g, "\n")); }

// ---------------------------------------------------------------- Phần 1: Hà, lý thuyết
function phan1(pres) {
  let sl = pres.addSlide();
  sl.background = { color: C.navy };
  sl.addImage({ path: path.join(GOC, "..", "hinh", "haui_Logo.jpg"), x: 0.7, y: 0.55, w: 1.0, h: 1.0, altText: "Logo HaUI" });
  sl.addText([{ text: "ĐẠI HỌC CÔNG NGHIỆP HÀ NỘI", options: { bold: true, breakLine: true } },
    { text: "Môn Khai phá dữ liệu và phát hiện tri thức · Mã lớp 20261IT7314001" }],
    { x: 1.95, y: 0.6, w: 10, h: 0.9, fontFace: BF, fontSize: 15, color: "CADCFC", margin: 0, valign: "middle" });
  sl.addText("Ứng dụng PSO kết hợp Fuzzy C-means trong phân cụm dữ liệu phục vụ đánh giá rủi ro tín dụng",
    { x: 0.7, y: 2.0, w: 11.9, h: 1.9, fontFace: HF, fontSize: 38, bold: true, color: C.white, margin: 0, valign: "middle" });
  sl.addText("Thử PSO tìm tâm khởi tạo, rồi chuyển sang PSO tìm trọng số thuộc tính cho FCM",
    { x: 0.7, y: 3.95, w: 11.9, h: 0.6, fontFace: BF, fontSize: 18, italic: true, color: "F2C48D", margin: 0 });
  sl.addText([{ text: "Giảng viên hướng dẫn: TS. Trần Hùng Cường", options: { breakLine: true } },
    { text: "Nhóm 1: Vương Quốc Bình · Trần Trường Giang · Nguyễn Quang Hà · Nguyễn Tiến Hải", options: { breakLine: true } },
    { text: "Hà Nội, 30/10/2026" }],
    { x: 0.7, y: 5.2, w: 11.9, h: 1.4, fontFace: BF, fontSize: 16, color: C.white, margin: 0, valign: "top",
      paraSpaceAfter: 6 });
  ghiChu(sl, `Kính chào Thầy và các bạn. Em là Nguyễn Quang Hà, đại diện nhóm 1. Tiểu luận của nhóm em là "Ứng dụng PSO kết hợp
    Fuzzy C-means trong phân cụm dữ liệu phục vụ đánh giá rủi ro tín dụng". Bài trình bày khoảng 50 phút, chia làm 3 phần. Em trình bày
    phần lý thuyết: bối cảnh, dữ liệu, thuật toán FCM và các câu hỏi nghiên cứu. Hải trình bày phần toán: 5 thuật toán PSO, các chỉ số
    đánh giá và 2 cách ghép PSO với FCM, gồm PSO tìm tâm và đề xuất PSO tìm trọng số. Giang trình bày toàn bộ thực nghiệm, từ TN1
    tới TN5, và kết luận.`);

  sl = trang(pres, 1, "Bối cảnh: phân khúc khách hàng theo rủi ro");
  const y0 = 1.65;
  [["Ngân hàng cần nhóm khách hàng theo rủi ro", "để quyết định lãi suất khoản vay, đặt hạn mức cho vay và chọn nhóm cần theo dõi sát."],
   ["Phân cụm không cần nhãn", "chia khách hàng thành vài phân khúc dễ giải thích; nhãn rủi ro lịch sử (vỡ nợ, khó khăn tài chính, phá sản tuỳ bộ) chỉ dùng để đánh giá."],
   ["Phân cụm mờ cho thêm độ thuộc", "nhận ra khách hàng \"biên\" nằm giữa 2 phân khúc; phân cụm cứng không có."]]
    .forEach(([t, d], i) => {
      soTron(pres, sl, 0.6, y0 + i * 1.65, 0.62, i + 1);
      sl.addText([{ text: t, options: { bold: true, color: C.navy, fontSize: 18, breakLine: true } },
        { text: d, options: { fontSize: 15, color: C.ink } }],
        { x: 1.45, y: y0 - 0.08 + i * 1.65, w: 6.0, h: 1.35, fontFace: BF, margin: 0, valign: "top" });
    });
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.9, y: 1.6, w: 4.85, h: 4.9, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  sl.addShape(pres.shapes.OVAL, { x: 8.2, y: 2.35, w: 2.6, h: 2.6, fill: { color: "2F7D5B", transparency: 75 },
    line: { color: C.green } });
  sl.addShape(pres.shapes.OVAL, { x: 9.85, y: 2.35, w: 2.6, h: 2.6, fill: { color: "C0392B", transparency: 75 },
    line: { color: "C0392B" } });
  sl.addText("Rủi ro thấp", { x: 8.3, y: 1.8, w: 2.2, h: 0.45, fontFace: BF, fontSize: 14, bold: true, color: C.green, margin: 0 });
  sl.addText("Rủi ro cao", { x: 10.4, y: 1.8, w: 2.1, h: 0.45, fontFace: BF, fontSize: 14, bold: true, color: "C0392B",
    align: "right", margin: 0 });
  sl.addShape(pres.shapes.OVAL, { x: 10.43, y: 3.5, w: 0.3, h: 0.3, fill: { color: C.navy }, line: { color: C.navy } });
  sl.addText([{ text: "Khách hàng biên", options: { bold: true, breakLine: true } },
    { text: "độ thuộc 0,55 vào nhóm thấp, 0,45 vào nhóm cao" }],
    { x: 8.15, y: 5.2, w: 4.4, h: 1.0, fontFace: BF, fontSize: 14, color: C.ink, align: "center", margin: 0 });
  ghiChu(sl, `Tổ chức tín dụng nào cũng cần chia khách hàng theo mức rủi ro, để quyết định lãi suất khoản vay, đặt hạn mức cho vay và biết nhóm nào cần
    giám sát. Phân cụm làm được việc này mà không cần nhãn, nên dùng được cả khi dữ liệu mới chưa có nhãn rủi ro lịch sử.
    Phân cụm cứng thì mỗi khách hàng bị gán hẳn vào đúng một cụm. Với phân cụm mờ, cụ thể là Fuzzy C-means, mình biết thêm độ thuộc của từng khách hàng vào từng cụm. Ví dụ một
    khách hàng có độ thuộc 0,55 vào nhóm rủi ro thấp và 0,45 vào nhóm rủi ro cao thì đó là khách hàng "biên", cần xem
    kỹ hơn. Nhóm chọn FCM vì có thông tin này.`);

  sl = trang(pres, 1, "Dữ liệu: 2 bộ chính");
  the(pres, sl, 0.6, 1.5, 5.95, 3.7, "Doanh nghiệp niêm yết Việt Nam", [
    "3.277 quan sát doanh nghiệp–năm, 825 công ty có cổ phiếu trên 3 sàn chứng khoán HOSE, HNX, UPCOM, giai đoạn 2010–2020.",
    "25 chỉ số tài chính: khả năng trả nợ ngắn hạn, mức vay nợ, kết quả kinh doanh, giá cổ phiếu, định giá công ty.",
    "Nhãn: gặp khó khăn tài chính ({soLieuMoTaVietNamNhanMot}%), gán theo công ty. Nguồn: Trần Kim Long và cộng sự (2022), DOI và GitHub ở slide sau."], { coThan: 16 });
  the(pres, sl, 6.8, 1.5, 5.95, 3.7, "Thẻ tín dụng Taiwan Default", [
    "30.000 chủ thẻ năm 2005, 23 thuộc tính: hạn mức thẻ, nhân khẩu học, tình trạng trả nợ 6 tháng, số tiền còn nợ trên sao kê, số tiền đã trả.",
    "Nhãn: vỡ nợ tháng kế tiếp ({soLieuMoTaTaiwanKiemChungNhanMot}%).",
    "2 mẫu phân tầng 5.000 khách hàng, không giao nhau: mẫu thiết kế và mẫu kiểm chứng."], { coThan: 16, mau: "DDEFE6" });
  sl.addText([{ text: "Tiền xử lý: ", options: { bold: true, color: C.navy } },
    { text: "làm sạch giá trị thiếu (số 0 giả ở X6, X7, X12), bỏ thuộc tính gần trùng, log có dấu, chuẩn hoá z-score." }],
    { x: 0.6, y: 5.45, w: 12.15, h: 1.2, fontFace: BF, fontSize: 15, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 6 });
  ghiChu(sl, `Nhóm dùng 2 bộ dữ liệu chính. Bộ đầu là dữ liệu dự báo khó khăn tài chính của doanh nghiệp niêm yết Việt Nam, gồm
    3.277 quan sát của 825 công ty với 25 chỉ số tài chính. Khoảng 13 phần trăm quan sát thuộc nhóm gặp khó khăn tài
    chính. Bộ thứ hai là thẻ tín dụng Taiwan Default, có 30.000 chủ thẻ, nhãn là vỡ nợ ở tháng kế tiếp, chiếm khoảng
    22 phần trăm. Nhóm lấy 2 mẫu 5.000 khách hàng không giao nhau để có một mẫu kiểm chứng độc lập. Mọi phương pháp đều qua cùng một bước tiền xử lý: chuẩn hoá z-score, tức đưa các cột về cùng thang đo bằng cách trừ trung bình rồi chia cho độ lệch chuẩn, và log có dấu, tức nén các giá trị cực lớn nhưng vẫn giữ dấu âm hoặc dương. Riêng bộ Việt Nam có số
    0 giả ở 3 cột, nhóm coi đó là giá trị thiếu.`);


  sl = trang(pres, 1, "Bộ Việt Nam: doanh nghiệp niêm yết");
  const nhomVn = [["Nhóm chỉ số", "Cột", "Ví dụ"],
    ["Thanh khoản", "X1–X3", "khả năng trả nợ ngắn hạn"],
    ["Đòn bẩy (mức vay nợ)", "X4–X9", "nợ so với vốn chủ sở hữu hoặc tổng tài sản"],
    ["Rủi ro kinh doanh", "X10–X13", "nợ nhà cung cấp, tài sản so với nợ, EBITDA so với nợ ngắn hạn cộng lãi vay"],
    ["Thị trường", "X14–X20", "giá cổ phiếu so với lợi nhuận, doanh thu...; vốn hoá"],
    ["Định giá", "X21–X25", "giá trị cả công ty (EV), lợi nhuận mỗi cổ phiếu (EPS)"]];
  sl.addTable(nhomVn.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i % 2 ? C.pale : C.white) } } }))),
    { x: 0.6, y: 1.5, w: 7.3, colW: [2.1, 1.3, 3.9], fontFace: BF, fontSize: 13, rowH: 0.62, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, margin: [3, 8, 3, 8] });
  sl.addText("3.277 quan sát doanh nghiệp–năm, 825 công ty (UPCOM 1.428 dòng, HNX 987, HOSE 862), 2010–2020, trừ tài chính, ngân hàng, bảo hiểm. Giải nghĩa thuật ngữ ở slide sau.",
    { x: 0.6, y: 5.3, w: 7.3, h: 0.7, fontFace: BF, fontSize: 13, italic: true, color: C.muted, margin: 0, valign: "top" });
  sl.addText([{ text: "Nguồn: ", options: { bold: true, color: C.navy } },
    { text: "Trần Kim Long, Lê Hoàng Anh, Nguyễn Thanh Hiền, Nguyễn Đức Trung (2022). Explainable Machine Learning for Financial Distress Prediction: Evidence from Vietnam. Data, 7(11), 160. ",
      options: { color: C.ink } },
    { text: "https://doi.org/10.3390/data7110160", options: { hyperlink: { url: "https://doi.org/10.3390/data7110160" }, color: "1F5FA8", underline: { style: "sng" } } },
    { text: "  ·  Dữ liệu: ", options: { color: C.ink } },
    { text: "https://github.com/anhle32/Explainable-Machine-Learning-", options: { hyperlink: { url: "https://github.com/anhle32/Explainable-Machine-Learning-" }, color: "1F5FA8", underline: { style: "sng" } } },
    { text: " (commit a7eb607, truy cập 13/10/2026).", options: { color: C.ink } }],
    { x: 0.6, y: 6.3, w: 12.15, h: 0.6, fontFace: BF, fontSize: 10.5, margin: 0, valign: "top" });
  the(pres, sl, 8.15, 1.5, 4.6, 2.35, "Nhãn: khó khăn tài chính", [
    "Gán 1 khi vốn chủ sở hữu âm, hoặc lợi nhuận trước lãi vay, thuế, khấu hao (EBITDA) thấp hơn tiền lãi vay 2 năm liền, hoặc lợi nhuận hoạt động âm 3 năm liền.",
    "436 trên 3.277 quan sát ({soLieuMoTaVietNamNhanMot}%); gán theo công ty (95 công ty), không theo năm."], { coThan: 12.5, coTieu: 15 });
  the(pres, sl, 8.15, 4.0, 4.6, 2.2, "Lưu ý khi xử lý", [
    "Số 0 giả ở X6, X7, X12 (khoảng 1.100 đến 1.300 dòng mỗi cột): coi là giá trị thiếu, điền trung vị.",
    "Bỏ X3, X15, X18, X21 vì gần trùng: còn {soLieuMoTaVietNamD} thuộc tính."], { coThan: 12.5, coTieu: 15, mau: "FCEFD9" });
  ghiChu(sl, `Em nói kỹ hơn về bộ Việt Nam, vì đây là bộ chính của nhóm. Bộ này có 25 chỉ số tài chính, chia thành 5 nhóm: khả năng trả nợ
    ngắn hạn, mức vay nợ so với vốn của chủ, rủi ro kinh doanh, giá cổ phiếu trên thị trường và định giá cả công ty. Dữ liệu gồm 3.277 quan sát doanh nghiệp–năm của 825 công ty
    niêm yết trên 3 sàn, giai đoạn 2010 đến 2020, không gồm tài chính, ngân hàng và bảo hiểm. Nhãn gặp khó khăn tài chính gán 1 khi
    công ty có vốn chủ sở hữu âm, hoặc lợi nhuận trước lãi vay, thuế và khấu hao nhỏ hơn tiền lãi vay 2 năm liên tiếp, nghĩa là hoạt động kinh doanh không đủ tiền trả lãi, hoặc lợi nhuận hoạt động âm 3 năm liên tiếp. Có 436
    quan sát mang nhãn này, khoảng 13 phần trăm, thuộc 95 công ty, và nhãn gán theo công ty chứ không theo từng năm. Đây là rủi ro
    tín dụng doanh nghiệp, khác các bộ tín dụng cá nhân. Một lưu ý khi xử lý: 3 cột X6, X7, X12 có số 0 giả thay cho giá trị
    thiếu, nên nhóm điền trung vị, và bỏ 4 cột gần trùng nên còn 21 thuộc tính. Nguồn của bộ này là bài báo của nhóm tác giả Trần Kim Long và cộng sự năm
    2022 trên tạp chí Data, có DOI ở cuối slide; tác giả công bố dữ liệu trên GitHub, nhóm tải ngày 13 tháng 10 năm 2026. Kho GitHub
    không ghi giấy phép riêng, bài báo thì theo CC BY 4.0.`);

  sl = trang(pres, 1, "Giải nghĩa thuật ngữ tài chính trong slide");
  const tuTrai = [["Thuật ngữ", "Nghĩa đơn giản"],
    ["Niêm yết; HOSE, HNX, UPCOM", "Cổ phiếu công ty được mua bán công khai trên sàn chứng khoán; 3 sàn ở TP.HCM, Hà Nội và sàn cho công ty đại chúng chưa niêm yết."],
    ["Vốn chủ sở hữu", "Phần tài sản thuộc về chủ công ty sau khi trừ nợ; âm nghĩa là nợ nhiều hơn tài sản."],
    ["Lãi vay", "Tiền lãi công ty phải trả cho các khoản đi vay."],
    ["Khấu hao", "Chi phí ghi nhận dần cho sự hao mòn của máy móc, nhà xưởng; không phải tiền chi ra thật."],
    ["EBITDA", "Lợi nhuận trước lãi vay, thuế và khấu hao: tiền hoạt động kinh doanh làm ra. Thấp hơn lãi vay nghĩa là không đủ tiền trả lãi."],
    ["EBIT", "Lợi nhuận trước lãi vay và thuế."],
    ["Thanh khoản, đòn bẩy", "Thanh khoản: có đủ tiền trả nợ ngắn hạn không. Đòn bẩy: đang vay nợ nhiều hay ít so với vốn của chủ; càng cao càng rủi ro."]];
  const tuPhai = [["Thuật ngữ", "Nghĩa đơn giản"],
    ["Vốn hoá", "Giá trị cả công ty trên sàn = giá một cổ phiếu × số cổ phiếu."],
    ["P/E, P/B, P/S, P/CF", "Giá cổ phiếu chia cho lợi nhuận (E), giá trị sổ sách (B), doanh thu (S), dòng tiền (CF) tính trên mỗi cổ phiếu."],
    ["EV (giá trị doanh nghiệp)", "Vốn hoá cộng nợ, trừ tiền mặt. EV/doanh thu, EV/EBITDA, EV/EBIT là EV chia cho từng đại lượng đó."],
    ["EPS", "Lợi nhuận trên mỗi cổ phiếu; \"pha loãng\" là tính cả cổ phiếu có thể phát hành thêm."],
    ["Hạn mức, dư nợ sao kê, PAY_x", "Hạn mức: mức chi tối đa của thẻ. Dư nợ sao kê: số tiền còn nợ. PAY_x: tình trạng trả nợ tháng x (số tháng trễ hạn)."],
    ["Vỡ nợ, phá sản, khó khăn tài chính", "Không trả được nợ đúng hạn; công ty ngừng hoạt động vì nợ; hoặc rơi vào các điều kiện của nhãn ở slide trước."],
    ["Doanh nghiệp–năm", "Mỗi dòng dữ liệu là một công ty trong một năm; một công ty xuất hiện nhiều dòng."]];
  const bangTu = (rows, x) => sl.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0,
    color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.navy : (i % 2 ? C.pale : C.white) } } }))),
    { x, y: 1.45, w: 6.0, colW: [1.75, 4.25], fontFace: BF, fontSize: 11.5, rowH: 0.55, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, margin: [3, 6, 3, 6] });
  bangTu(tuTrai, 0.6); bangTu(tuPhai, 6.75);
  ghiChu(sl, `Slide này giải nghĩa các từ kinh doanh và tài chính, vì nhiều bạn học lập trình nên có thể chưa quen. Doanh nghiệp niêm yết là công
    ty có cổ phiếu được mua bán công khai trên sàn chứng khoán, ở Việt Nam có 3 sàn là HOSE, HNX và UPCOM. Vốn chủ sở hữu là phần tài sản thuộc về chủ công
    ty sau khi trừ hết nợ; nếu âm nghĩa là nợ nhiều hơn tài sản. Lãi vay là tiền lãi phải trả cho các khoản đi vay. Khấu hao là chi phí ghi nhận dần cho sự
    hao mòn của máy móc, nhà xưởng, không phải tiền chi ra thật. EBITDA là lợi nhuận trước lãi vay, thuế và khấu hao, hiểu đơn giản là số tiền mà hoạt
    động kinh doanh làm ra; nếu nó thấp hơn tiền lãi vay thì công ty không đủ tiền trả lãi. Thanh khoản là khả năng có đủ tiền trả nợ ngắn hạn, còn đòn
    bẩy là mức vay nợ so với vốn của chủ. Vốn hoá là giá trị cả công ty trên sàn. Các tỉ số P/E, P/B, P/S, P/CF là giá cổ phiếu chia cho lợi nhuận, giá trị
    sổ sách, doanh thu và dòng tiền tính trên mỗi cổ phiếu. EV là giá trị doanh nghiệp, bằng vốn hoá cộng nợ trừ tiền mặt. EPS là lợi nhuận trên mỗi cổ phiếu.
    Với thẻ tín dụng, hạn mức là mức chi tối đa của thẻ, dư nợ là số tiền còn nợ, và các cột PAY cho biết từng tháng khách trả nợ trễ hay đúng hạn.`);

  sl = trang(pres, 1, "7 bộ dữ liệu thực nghiệm");
  const uci = (id) => ({ ten: `UCI ${id}`, url: `https://archive.ics.uci.edu/dataset/${id}` });
  const bo = [["Việt Nam (niêm yết)", "Khó khăn tài chính doanh nghiệp", "3.277", "VietNam", "TN1–TN5",
      { lien: [{ ten: "DOI", url: "https://doi.org/10.3390/data7110160" }, { ten: "GitHub", url: "https://github.com/anhle32/Explainable-Machine-Learning-" }] }],
    ["Taiwan Default", "Vỡ nợ thẻ tín dụng", "5.000", "TaiwanKiemChung", "TN1–TN5", uci(350)],
    ["South German Credit", "Tín dụng cá nhân", "1.000", "SouthGerman", "TN2, TN3", uci(573)],
    ["German Credit (bản số)", "Tín dụng cá nhân", "1.000", "GermanSo", "TN2, TN3", uci(144)],
    ["Australian Credit", "Duyệt hồ sơ tín dụng", "690", "Australian", "TN2, TN3", uci(143)],
    ["Polish Bankruptcy", "Phá sản doanh nghiệp", "5.000", "Polish", "TN2, TN3", uci(365)],
    ["Taiwan Bankruptcy", "Phá sản doanh nghiệp", "5.000", "TaiwanPhaSan", "TN2, TN3", uci(572)]];
  const bangBo = [["Bộ dữ liệu", "Loại rủi ro", "n", "D", "Tỉ lệ nhãn rủi ro (%)", "Dùng ở", "Nguồn"]].concat(bo.map(([ten, loai, nn, k, tn, nguon]) =>
    [ten, loai, nn, n(`soLieuMoTa${k}D`), n(`soLieuMoTa${k}NhanMot`), tn, nguon]));
  sl.addTable(bangBo.map((r, i) => r.map((c, j) => (j === 6 && i > 0
    ? { text: (c.lien || [c]).flatMap((l, k) => [
        ...(k ? [{ text: " · ", options: { color: C.ink } }] : []),
        { text: l.ten, options: { hyperlink: { url: l.url, tooltip: l.url }, color: "1F5FA8", underline: { style: "sng" } } }]),
      options: { fill: { color: i <= 2 ? C.tint : (i % 2 ? C.pale : C.white) } } }
    : { text: c, options: { bold: i === 0 || (j === 0 && i <= 2),
    align: j >= 2 && j <= 4 ? "center" : "left", color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i <= 2 ? C.tint : (i % 2 ? C.pale : C.white)) } } }))),
    { x: 0.6, y: 1.5, w: 12.15, colW: [2.45, 2.95, 0.85, 0.7, 1.25, 1.35, 2.6], fontFace: BF, fontSize: 13.5, rowH: 0.55,
      valign: "middle", border: { type: "solid", pt: 0.75, color: C.line }, margin: [3, 8, 3, 8] });
  sl.addText("2 dòng đầu là bộ chính. Taiwan Default và 2 bộ phá sản lấy mẫu phân tầng 5.000 quan sát; n là số quan sát sau lấy mẫu, D là số thuộc tính sau tiền xử lý. Nhãn rủi ro: khó khăn tài chính, vỡ nợ, vay xấu, hồ sơ bị từ chối hoặc phá sản tuỳ bộ. UCI: CC BY 4.0, truy cập 13/10/2026.",
    { x: 0.6, y: 6.05, w: 12.15, h: 0.5, fontFace: BF, fontSize: 12, italic: true, color: C.muted, margin: 0 });
  ghiChu(sl, `Ngoài 2 bộ chính, nhóm dùng thêm 5 bộ dữ liệu rủi ro công khai cho TN2 và TN3, gồm 3 bộ tín dụng cá nhân và 2 bộ phá sản doanh nghiệp. Ba bộ đầu là tín dụng cá nhân cỡ nhỏ:
    South German, German bản số và Australian, đa số thuộc tính là mã hoặc thứ bậc. Hai bộ sau là dữ liệu phá sản doanh
    nghiệp, không phải hồ sơ vay, nhưng phá sản là dạng vỡ nợ nặng nhất của doanh nghiệp nên nhóm dùng làm dữ liệu thay thế cho rủi ro tín dụng doanh nghiệp; các bộ này nhiều chiều, Polish và Taiwan Bankruptcy, nhãn lệch mạnh, chỉ khoảng 3 đến 5 phần trăm. Các bộ phụ dùng để xem kết quả có khái quát được ra ngoài 2 bộ chính không. Trên các bộ dữ liệu này, nhóm dùng thuật toán FCM.`);

  sl = trang(pres, 1, "Fuzzy C-means và 2 điểm yếu");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.55, w: 12.15, h: 1.55, fill: { color: C.pale },
    line: { color: C.pale }, rectRadius: 0.1 });
  anh(sl, CT(1), 1.0, 1.7, { w: 6.6, h: 1.25 });
  sl.addText("FCM luân phiên cập nhật độ thuộc U và tâm V để giảm Jm; m = 2 là giá trị thường dùng.",
    { x: 8.0, y: 1.7, w: 4.5, h: 1.25, fontFace: BF, fontSize: 14, color: C.ink, margin: 0, valign: "middle" });
  the(pres, sl, 0.6, 3.45, 5.9, 3.2, "Điểm yếu 1: phụ thuộc khởi tạo",
    ["Kết quả có thể đổi theo tâm ban đầu và dừng ở cực tiểu địa phương.",
     "Các bài PSO–FCM đã đọc dùng PSO tìm tâm khởi tạo để sửa điểm này.",
     "Giai đoạn 1 của nhóm kiểm tra cách làm đó."], { so: 1 });
  the(pres, sl, 6.85, 3.45, 5.9, 3.2, "Điểm yếu 2: mọi thuộc tính nặng như nhau",
    ["Khoảng cách Euclid coi mọi thuộc tính ngang nhau, cột thừa hay nhiễu cũng kéo cụm.",
     "FCM không tự học được trọng số: cực tiểu Jm theo trọng số dồn hết vào một thuộc tính.",
     "Giai đoạn 2 dùng PSO tìm trọng số."], { so: 2, mau: "FCEFD9" });
  ghiChu(sl, `Ở đây c là số cụm và m là hệ số độ mờ, m bằng 2 là giá trị mặc định. FCM tìm cực tiểu của hàm mục tiêu Jm, là tổng bình phương khoảng cách từ mỗi điểm tới từng tâm, nhân với độ thuộc mũ
    m. Tài liệu hay nhắc tới 2 điểm yếu của thuật toán này. Điểm yếu đầu là kết quả phụ thuộc tâm khởi tạo, vì thuật toán có thể mắc kẹt ở một cực tiểu địa phương, tức điểm thấp cục bộ chưa chắc là thấp nhất, và các
    bài PSO–FCM nhóm đọc đều dùng PSO để tìm tâm khởi tạo tốt hơn. Điểm yếu thứ hai là FCM đo khoảng cách Euclid, tức khoảng cách đường thẳng thông thường, nên coi mọi thuộc tính nặng như nhau mà lại không tự học được trọng số: nếu tối ưu Jm theo trọng số thì nghiệm dồn hết trọng số vào một thuộc
    tính. Nhóm làm 2 giai đoạn, mỗi giai đoạn nhắm vào một điểm yếu.`);

  sl = trang(pres, 1, "10 bài báo đã khảo sát (2021–2025)");
  const muoi = [
    ["01 Cui 2022", "Phát hiện hành vi bất thường", "PSO thay FCM, fitness cosine", "NSL-KDD", "Tổng quan", "Giang"],
    ["02 Qian 2022", "Chọn tổ hợp quan sát BeiDou-3", "PSO thay FCM, 3 lịch ω", "Iris, BDS-3", "PSO-S, PSO-LOG", "Hà"],
    ["03 Shang 2021", "Dự báo phụ tải điện", "PSO rồi kernel FCM", "Phụ tải NSW", "Tổng quan", "Hà"],
    ["04 Zhang 2025", "Phân loại làn điệu hí khúc", "QPSO rồi FCM", "Chuỗi cao độ", "QPSO", "Hải"],
    ["05 Sun 2024", "Cải tiến FCM tổng quát", "Bầy cá ngừ, cá thể là ma trận U", "2 bộ nhân tạo, 5 bộ UCI", "Tổng quan", "Hải"],
    ["06 Liu 2021", "Cải tiến FCM có trọng số", "GWO tìm tâm WFCM", "6 bộ UCI", "Trọng số thuộc tính", "Hải"],
    ["07 Bosker 2025", "Dự báo LGD theo cụm", "Không có metaheuristic", "Khoản vay vỡ nợ", "Bối cảnh tín dụng", "Bình"],
    ["08 Vasheghani 2025", "Phân khúc, dự báo vỡ nợ", "K-means; GA–PSO chỉnh mạng nơ-ron", "Ngân hàng Iran", "Bối cảnh tín dụng", "Bình"],
    ["09 Lakra 2025", "Chấm điểm tín dụng", "PSO chỉnh XGBoost", "German, Taiwan, P2P", "Bối cảnh tín dụng", "Giang"],
    ["10 Valdez 2021", "Tổng quan trắc lượng", "Tối ưu (m, c) theo chỉ số hợp lệ", "Web of Science", "Tổng quan", "Giang"]];
  const nhomMau = (i) => (i <= 6 ? C.tint : (i <= 9 ? "FCEFD9" : C.pale));
  const bangMuoi = [["Bài", "Bài toán", "Cách kết hợp", "Dữ liệu", "Dùng trong tiểu luận", "Người đọc"]].concat(muoi);
  sl.addTable(bangMuoi.map((r, i) => r.map((c, j) => ({ text: c, options: {
    bold: i === 0 || j === 0 || (j === 4 && i > 0 && i <= 6 && c !== "Tổng quan"),
    color: i === 0 ? C.white : C.ink, align: j === 5 ? "center" : "left",
    fill: { color: i === 0 ? C.navy : nhomMau(i) } } }))),
    { x: 0.6, y: 1.4, w: 12.15, colW: [1.95, 2.55, 2.75, 1.95, 1.95, 1.0], fontFace: BF, fontSize: 11.5, rowH: 0.4,
      valign: "middle", border: { type: "solid", pt: 0.75, color: C.line }, margin: [2, 6, 2, 6] });
  sl.addText("Xanh: FCM kết hợp metaheuristic, không bài nào dùng dữ liệu tín dụng · Cam: 3 bài tín dụng, không dùng phân cụm mờ · Không bài nào đo thời gian của bước phân cụm hay cho FCM cùng ngân sách tính toán.",
    { x: 0.6, y: 6.0, w: 12.15, h: 0.75, fontFace: BF, fontSize: 13, italic: true, color: C.muted, margin: 0, valign: "top" });
  ghiChu(sl, `Chương 1 của tiểu luận tổng hợp từ 10 bài báo trong 5 năm gần đây, mỗi thành viên đọc 2 đến 3 bài; cột cuối ghi người
    đọc, nên Thầy hỏi về bài nào thì bạn đọc bài đó trả lời. Sáu bài đầu, tô xanh, ghép FCM với một thuật toán bầy đàn hay
    metaheuristic, nhưng không bài nào dùng dữ liệu tín dụng. Ba bài tô cam là bài tín dụng, nhưng lại không dùng phân cụm mờ.
    Bài 10 là tổng quan các phương pháp phân cụm mờ lấy cảm hứng sinh học. Từ các bài này, nhóm lấy 3 thuật toán cho Chương 2:
    PSO-S và PSO-LOG từ bài 02, QPSO từ bài 04, còn ý tưởng trọng số thuộc tính lấy từ bài 06. Điểm chung nhóm nhận ra là
    không bài nào đo thời gian của bước phân cụm, và đối chứng FCM thuần thường thiếu hoặc không cùng điều kiện, nên chưa rõ
    PSO thật sự giúp FCM đến đâu. Đó là lý do nhóm làm 2 giai đoạn.`);

  sl = trang(pres, 1, "2 giai đoạn nghiên cứu");
  const bx = [0.6, 4.75, 8.9];
  const khoi = [["Giai đoạn 1", "PSO tìm tâm khởi tạo cho FCM", "Cách ghép phổ biến nhất trong các bài đã đọc.", C.tint],
    ["Phát hiện", "Không cải thiện FCM", "Jm hầu như chỉ có 1 cực tiểu; k-means++ đã đủ, PSO chậm hơn.", "FCEFD9"],
    ["Giai đoạn 2", "PSO tìm trọng số thuộc tính", "Thứ FCM không tự tối ưu được; đây là đề xuất của nhóm.", "DDEFE6"]];
  khoi.forEach(([a, b, c, mau], i) => {
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx[i], y: 2.0, w: 3.75, h: 3.6, fill: { color: mau }, line: { color: mau },
      rectRadius: 0.12 });
    sl.addText([{ text: a, options: { fontSize: 14, color: C.muted, bold: true, breakLine: true } },
      { text: b, options: { fontSize: 22, color: C.navy, bold: true, fontFace: HF, breakLine: true } },
      { text: " ", options: { fontSize: 8, breakLine: true } },
      { text: c, options: { fontSize: 15, color: C.ink } }],
      { x: bx[i] + 0.3, y: 2.25, w: 3.15, h: 3.1, fontFace: BF, margin: 0, valign: "top" });
    if (i < 2) muiTen(pres, sl, bx[i] + 3.8, 3.6, 0.35);
  });
  sl.addText("Thứ tự trình bày: Hà (lý thuyết), Hải (thuật toán và mô hình), Giang (thực nghiệm TN1–TN5 và kết luận).",
    { x: 0.6, y: 6.0, w: 12.1, h: 0.5, fontFace: BF, fontSize: 14, italic: true, color: C.muted, margin: 0 });
  ghiChu(sl, `Ở giai đoạn 1, nhóm làm theo đúng các bài báo, tức là dùng PSO tìm tâm khởi tạo cho FCM. Kết quả không như nhóm mong
    đợi: phiên bản PSO nào cũng cho cùng nghiệm với FCM khởi tạo k-means++, mà lại chậm hơn. K-means++ là cách chọn các tâm ban đầu sao cho rải xa nhau, rất rẻ và chạy nhanh. Tìm nguyên nhân thì nhóm
    thấy hàm mục tiêu trên dữ liệu tín dụng hầu như chỉ có 1 cực tiểu, nên PSO không còn gì để cải thiện. Vì vậy
    sang giai đoạn 2, nhóm dùng PSO cho thứ FCM không tự tối ưu được, đó là trọng số thuộc tính.`);

  sl = trang(pres, 1, "4 câu hỏi nghiên cứu và 5 thí nghiệm");
  const ch = [["CH1 · PSO tìm tâm", "Tâm do PSO tìm có làm FCM tốt hơn khởi tạo ngẫu nhiên, k-means++ không; phụ thuộc số cực tiểu ra sao?", "TN1, TN2"],
    ["CH2 · PSO tìm trọng số", "Phân khúc có phân biệt rủi ro cao và thấp rõ hơn FCM thường và trọng số tính bằng công thức không; lợi ích có do cơ chế bầy đàn (nhiều hạt cùng tìm) không?", "TN3"],
    ["CH3 · 5 phiên bản PSO", "PSO gốc, PSO-W, PSO-S, PSO-LOG, QPSO khác nhau thế nào khi tìm trọng số, ở 2 mức ngân sách?", "TN4"],
    ["CH4 · Kiểm chứng", "Trọng số học trên một mẫu có giữ lợi ích trên mẫu độc lập không; khi nào không có ích?", "TN5"]];
  ch.forEach(([t, d, ai], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.55 + Math.floor(i / 2) * 2.6;
    the(pres, sl, x, y, 5.95, 2.4, t, d, { so: i + 1, coThan: 16 });
    sl.addText(ai, { x: x + 3.4, y: y + 1.95, w: 2.35, h: 0.35, fontFace: BF, fontSize: 12.5, bold: true, color: C.amber,
      align: "right", margin: 0 });
  });
  ghiChu(sl, `Từ 2 giai đoạn đó, nhóm đặt ra 4 câu hỏi. CH1 là PSO tìm tâm có giúp FCM không; câu này trả lời bằng TN1 và TN2. CH2 hỏi: khi PSO tìm trọng số thì các cụm có phân biệt được khách hàng rủi ro cao và rủi ro thấp rõ hơn không, và lợi ích đó có phải nhờ cơ chế bầy đàn, tức nhiều hạt cùng tìm kiếm, không; câu này trả lời bằng TN3. CH3 so 5 phiên bản PSO ở TN4. CH4 kiểm tra trọng số
    trên mẫu độc lập ở TN5. Cả 5 thí nghiệm Giang sẽ trình bày sau phần thuật toán. Em mời Hải trình bày phần thuật toán.`);
}

// ---------------------------------------------------------------- Phần 2: Hải, thuật toán và mô hình
function phanNen(pres) {
  let sl = trang(pres, 2, "PSO gốc và PSO-W");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 12.15, h: 1.95, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(2), 0.9, 1.6, { w: 11.5, h: 1.75 });
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 3.65, w: 12.15, h: 1.05, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(3), 0.9, 3.75, { w: 8.0, h: 0.85 });
  sl.addText("PSO-W", { x: 9.3, y: 3.75, w: 3.2, h: 0.85, fontFace: HF, fontSize: 20, bold: true, color: C.navy, margin: 0,
    valign: "middle", align: "right" });
  gach(sl, ["Δx là bước dịch chuyển của hạt sau mỗi vòng (\"vận tốc\"): một phần bước cũ, phần kéo về pbest, phần kéo về gbest.",
    "PSO gốc (Kennedy, Eberhart 1995): ω = 1, c₁ = c₂ = 2, bước bị chặn ở 0,2 lần độ rộng hộp tìm kiếm.",
    "PSO-W (Shi, Eberhart 1998): ω giảm tuyến tính, khám phá rộng lúc đầu, khai thác quanh gbest lúc cuối; cũng là đối chứng L-PSO của bài 02."],
    { x: 0.6, y: 4.95, w: 12.1, h: 1.9 }, 15);
  ghiChu(sl, `Em là Hải. Em trình bày 5 thuật toán PSO nhóm so sánh, các chỉ số đánh giá, rồi 2 cách ghép PSO với FCM: tìm tâm khởi tạo và tìm trọng số thuộc tính. PSO là thuật toán bầy đàn: mỗi hạt là một lời giải ứng viên, ví dụ một bộ tâm cụm, và cả bầy cùng tìm lời giải tốt nhất. Fitness là điểm số đo một lời giải tốt đến đâu, ở đây là hàm mục tiêu cần giảm. Em đi lần lượt từ bản gốc. Ở PSO gốc, bước dịch chuyển của mỗi hạt được cập nhật theo từng chiều. Bước mới giữ một phần bước cũ, cộng thêm phần kéo về vị trí tốt nhất mà chính hạt từng đạt, gọi là pbest, và phần kéo về vị trí tốt nhất mà cả bầy từng đạt, gọi là gbest, kèm hệ số ngẫu nhiên r. Lấy vị trí
    cũ cộng bước này là ra vị trí mới. Omega là hệ số quán tính, quyết định hạt giữ lại bao nhiêu bước cũ. Bản gốc không có tham số omega riêng, tức là omega bằng 1: hạt giữ nguyên toàn bộ bước cũ, quán tính ở mức cao nhất và không giảm theo vòng, nên bước dễ lớn dần và phải chặn lại. PSO-W cho omega giảm tuyến tính từ 0,9 xuống 0,4, nên lúc đầu bầy đi xa, về cuối thì co lại quanh gbest. Bài
    02 cũng lấy lịch tuyến tính này làm đối chứng.`);

  sl = trang(pres, 2, "PSO-S và PSO-LOG: lịch quán tính phi tuyến (bài 02)");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 6.0, h: 2.0, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(4), 0.8, 1.6, { w: 5.6, h: 1.8 });
  gach(sl, ["Lấy từ Qian và cộng sự 2022 (bài 02), giữ nguyên tham số: ωs = 1,2; ωe = 0,3; z = 1,51.",
    "PSO-S giữ ω cao lâu rồi giảm nhanh về cuối; PSO-LOG giảm chậm, hẹp (khoảng 0,87 → 0,61).",
    "Bài 02 kết luận S tốt nhất, nhưng Bảng 5 của bài: LOG dẫn đầu ở 300 vòng."], { x: 0.6, y: 3.75, w: 6.0, h: 2.9 }, 15);
  const t = Array.from({ length: 21 }, (_, i) => i / 20);
  sl.addChart(pres.charts.LINE, [
    { name: "PSO-W", labels: t.map((v, i) => (i % 5 ? "" : v.toFixed(2).replace(".", ","))), values: t.map((v) => 0.9 - 0.5 * v) },
    { name: "PSO-S", labels: t.map((v, i) => (i % 5 ? "" : v.toFixed(2).replace(".", ","))), values: t.map((v) => 1.2 - 0.9 * v * v) },
    { name: "PSO-LOG", labels: t.map((v, i) => (i % 5 ? "" : v.toFixed(2).replace(".", ","))), values: t.map((v) => Math.pow(Math.log(2.98 + v), -1.51)) }],
    { x: 6.9, y: 1.5, w: 5.85, h: 5.1, chartColors: ["9AA9C2", C.amber, C.navy], lineSize: 2.5, lineDataSymbol: "none",
      showLegend: true, ...FONT_BD, legendPos: "b", legendFontSize: 12, catAxisLabelFontSize: 10, valAxisLabelFontSize: 11,
      valAxisMinVal: 0.2, valAxisMaxVal: 1.3, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "ω theo tiến độ t/T", titleFontSize: 13, titleColor: C.navy });
  ghiChu(sl, `Hai phiên bản tiếp theo lấy từ bài 02 của Qian và cộng sự, và cả hai chỉ đổi lịch giảm của omega. Với
    PSO-S, omega giảm theo bình phương, nên giữ ở mức cao khá lâu rồi mới giảm nhanh về cuối. PSO-LOG giảm theo hàm
    logarit trong một khoảng hẹp, từ khoảng 0,87 xuống 0,61. Biểu đồ bên phải vẽ cả 3 lịch để so. Nhóm giữ nguyên
    tham số như bài 02. Bài 02 kết luận PSO-S tốt nhất, nhưng bảng số liệu của chính bài cho thấy ở 300 vòng LOG tốt nhất,
    S chỉ dẫn đầu ở 1.000 vòng.`);

  sl = trang(pres, 2, "QPSO: PSO hành vi lượng tử (bài 04)");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 12.15, h: 2.35, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(5), 1.0, 1.65, { w: 11.3, h: 2.05 });
  gach(sl, ["Hạt không có vận tốc: mỗi vòng được lấy mẫu quanh điểm hút φ nằm giữa pbest và gbest.",
    "Độ rộng bước tỉ lệ với khoảng cách tới trung bình các pbest (mbest), nhân hệ số γ giảm từ 1,0 xuống 0,1.",
    "Dạng lấy từ Zhang và cộng sự 2025 (bài 04); công thức 17 của bài in sai, nhóm sửa theo QPSO chuẩn của Sun 2004."],
    { x: 0.6, y: 4.15, w: 12.1, h: 2.6 }, 16);
  ghiChu(sl, `Phiên bản cuối là QPSO, lấy từ bài 04. QPSO khác các phiên bản còn lại
    ở chỗ không có vận tốc. Mỗi vòng, vị trí mới của hạt được rút ngẫu nhiên quanh điểm hút phi, nằm giữa pbest của
    hạt và gbest của cả bầy. Độ rộng bước tỉ lệ với khoảng cách từ hạt tới mbest, là trung bình các pbest, nhân với
    gamma giảm dần từ 1 xuống 0,1. Bước nhảy vì thế thu hẹp dần, giống omega giảm dần ở PSO-W. Công thức 17 trong bài 04 bị
    in sai: gbest không xuất hiện trong công thức, nhóm đã sửa theo bài gốc của Sun năm 2004.`);

  sl = trang(pres, 2, "Chỉ số đánh giá và cách kiểm định");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 12.15, h: 1.3, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(8), 0.9, 1.6, { w: 11.5, h: 1.1 });
  the(pres, sl, 0.6, 3.05, 3.9, 3.55, "Chỉ số chính", ["Chênh tỉ lệ nhãn rủi ro giữa các cụm Δr: càng lớn, các cụm càng khác nhau về rủi ro.",
    "AUC phân hoạch: xác suất một khách hàng có nhãn rủi ro nằm ở cụm rủi ro cao hơn một khách hàng không có.",
    "Dùng nhãn chỉ để đánh giá; dữ liệu mới chưa có nhãn thì phân cụm vẫn chạy."], { coThan: 15.5 });
  the(pres, sl, 4.73, 3.05, 3.9, 3.55, "Chỉ số phụ", ["Silhouette: chính là fitness, nên không dùng để chứng minh.",
    "PC, XB về độ mờ và độ tách; thời gian và số lần tính fitness."], { coThan: 14, mau: "DDEFE6" });
  the(pres, sl, 8.85, 3.05, 3.9, 3.55, "Kiểm định", ["Wilcoxon hai phía theo cặp seed (n = 30).",
    "Holm trên toàn họ phép so của mỗi thí nghiệm, α = 0,05.",
    "Không có ý nghĩa thì viết \"không phát hiện khác biệt\"."], { coThan: 14, mau: "FCEFD9" });
  ghiChu(sl, `Mục tiêu là phân khúc theo rủi ro, nên 2 chỉ số chính đo đúng việc đó. Chỉ số đầu là chênh tỉ lệ nhãn rủi ro giữa cụm cao nhất và cụm thấp nhất, ví dụ cụm này có 50 phần trăm doanh nghiệp gặp khó khăn còn cụm kia chỉ 5 phần trăm; chênh càng lớn thì phân khúc càng phân biệt được rủi ro. Nhãn này là vỡ nợ ở Taiwan Default, khó khăn tài chính ở bộ Việt Nam, phá sản ở các bộ
    phụ. Chỉ số thứ hai là AUC của phân hoạch, một con số từ 0,5, tức phân biệt rủi ro như đoán mò, tới 1, tức phân biệt hoàn hảo; cụ thể nó là tức xác suất một khách hàng có nhãn rủi ro rơi vào cụm rủi ro cao hơn
    một khách hàng không có. Hai chỉ số này dùng nhãn chỉ để đánh giá, còn khi dữ liệu mới chưa có nhãn thì phân cụm vẫn chạy được. Silhouette là chỉ số đo các cụm có chặt và tách xa nhau không, không cần nhãn, và chỉ là chỉ số phụ, vì nó chính là fitness của PSO. Mỗi seed là một lần chạy lặp lại với số khởi tạo ngẫu nhiên khác, nhóm chạy 30 seed. Mọi phép so đều là Wilcoxon theo cặp seed, tức kiểm định so hai cách trên cùng 30 lần chạy mà không cần giả định dữ liệu theo phân phối chuẩn, có hiệu chỉnh Holm để khi so nhiều cặp cùng lúc không bị báo nhầm là có khác biệt. Khi không
    có ý nghĩa, nhóm chỉ nói "không phát hiện khác biệt", chứ không nói hai bên ngang nhau.`);

}

// ---------------------------------------------------------------- Phần 2 (tiếp): Hải, PSO tìm tâm
function phan2(pres) {
  let sl = trang(pres, 2, "PSO tìm tâm khởi tạo: cách làm và thí nghiệm TN1");
  const buoc = [["Hạt = bộ c tâm", "mỗi hạt mã hoá c × D số"], ["Fitness = Jm", "đánh giá bộ tâm bằng hàm mục tiêu FCM"],
    ["gbest làm tâm đầu", "PSO chạy N hạt × T vòng"], ["FCM chạy tiếp", "tới hội tụ, ra nghiệm cuối"]];
  buoc.forEach(([a, b], i) => {
    const x = 0.6 + i * 3.1;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 2.7, h: 1.7, fill: { color: i === 3 ? "DDEFE6" : C.tint },
      line: { color: C.tint }, rectRadius: 0.1 });
    sl.addText([{ text: a, options: { bold: true, fontSize: 16, color: C.navy, breakLine: true } },
      { text: b, options: { fontSize: 13.5, color: C.ink } }], { x: x + 0.2, y: 1.75, w: 2.3, h: 1.4, fontFace: BF,
      margin: 0, valign: "middle", align: "center" });
    if (i < 3) muiTen(pres, sl, x + 2.72, 2.24, 0.36);
  });
  the(pres, sl, 0.6, 3.75, 6.0, 2.9, "Thiết kế TN1", ["5 phiên bản: PSO gốc, PSO-W, PSO-S, PSO-LOG, QPSO.",
    "Lưới N = 10, 20, 30 hạt × T = 20, 30, 50 vòng, 30 seed.",
    "Đối chứng: FCM khởi tạo U ngẫu nhiên, khởi tạo từ c điểm dữ liệu, k-means++."], { coThan: 16 });
  the(pres, sl, 6.85, 3.75, 5.9, 2.9, "2 câu hỏi đo riêng", ["PSO tối ưu Jm tốt đến đâu (độ lệch gbest so với Jm tốt nhất)?",
    "Sau khi FCM chạy, nghiệm cuối có tốt hơn k-means++ không?"], { mau: "FCEFD9", coThan: 16 });
  ghiChu(sl, `Ở giai đoạn 1, nhóm dùng 5 phiên bản PSO vừa giới thiệu để tìm tâm khởi tạo cho FCM. Mỗi hạt là một bộ c tâm, còn fitness chính là hàm Jm của FCM. PSO chạy xong thì lấy gbest làm tâm
    khởi tạo, rồi cho FCM chạy tiếp tới khi hội tụ, tức lặp đến lúc kết quả không đổi nữa. TN1 thử 5 phiên bản trên lưới số hạt 10, 20, 30 và số vòng 20,
    30, 50, mỗi ô chạy 30 seed, rồi so với 3 cách khởi tạo không dùng PSO. Nhóm tách ra đo 2 việc: PSO tối ưu có
    tốt không, và nghiệm cuối sau FCM có tốt hơn không. Kết quả TN1 Giang sẽ trình bày ở phần thực nghiệm.`);

}

// ---------------------------------------------------------------- Phần 3: Giang, TN1 và TN2
function phanTN12(pres) {
  let sl = trang(pres, 3, "TN1: các bản cải tiến tối ưu tốt hơn PSO gốc…");
  anh(sl, ANH("e7-quy-dao-gbest.png"), 0.6, 1.5, { w: 7.6, h: 3.4 });
  sl.addText("Độ lệch gbest so với Jm tốt nhất qua các vòng PSO (m = 2, c = 3; trung vị 30 seed, dải IQR).",
    { x: 0.6, y: 4.95, w: 7.6, h: 0.4, fontFace: BF, fontSize: 11.5, italic: true, color: C.muted, margin: 0 });
  soLon(sl, 8.6, 1.5, 4.1, "26,0% → 22,5%", "PSO gốc: chỉ giảm khi tăng số hạt (10 → 30 hạt, Taiwan Default)", C.navy, 32);
  soLon(sl, 8.6, 3.35, 4.1, "9,5% · 10,1%", "PSO-W và QPSO, 30 hạt × 50 vòng: bám cực tiểu sát nhất", C.green);
  sl.addText("Ở ô tốt nhất, gbest vẫn cao hơn Jm tốt nhất khoảng 6–10%: phần còn lại do FCM làm.",
    { x: 0.6, y: 5.6, w: 12.1, h: 0.8, fontFace: BF, fontSize: 16, color: C.ink, margin: 0 });
  ghiChu(sl, `Em là Giang. Em trình bày toàn bộ thực nghiệm, bắt đầu với TN1 mà Hải vừa giới thiệu cách làm. Nếu chỉ coi PSO là thuật toán tối ưu thì các bản cải tiến đúng là tốt hơn PSO gốc. Trên biểu đồ, dải IQR là khoảng giữa phân vị 25 và 75 của 30 lần chạy, cho biết độ dao động. PSO gốc giữ quán tính cố định bằng 1, không giảm dần theo vòng, nên độ lệch chỉ giảm khi thêm hạt, từ 26 xuống 22,5 phần trăm. PSO-W và QPSO với 30 hạt, 50 vòng thì xuống
    khoảng 10 phần trăm. Nhưng ngay ở ô tốt nhất, gbest vẫn cao hơn cực tiểu khoảng 6 đến 10 phần trăm, tức là phần
    còn lại vẫn do FCM làm. Vậy sau khi FCM chạy xong thì sao?`);

  sl = trang(pres, 3, "…nhưng sau khi FCM chạy, mọi cách cho cùng nghiệm");
  soLon(sl, 0.6, 1.55, 3.9, "30/30 seed", "mọi phiên bản, mọi ô lưới đạt Jm tốt nhất (2 bộ chính, c = 3), giống FCM k-means++");
  soLon(sl, 4.75, 1.55, 3.9, "{soLieuENamVietNamCBaTiLeThoiGianTv}–{soLieuENamTaiwanKiemChungCBaTiLeThoiGianTv} lần", "PSO + FCM chậm hơn FCM k-means++ (Việt Nam, Taiwan Default)", C.amber);
  soLon(sl, 8.9, 1.55, 3.85, "77–100%", "Taiwan Default chưa log, c = 6 (2 cực tiểu): tỉ lệ đạt không theo phiên bản hay ngân sách");
  the(pres, sl, 0.6, 3.75, 12.15, 2.9, "Đọc kết quả", ["Thứ hạng tối ưu của các phiên bản không chuyển thành kết quả phân cụm.",
    "So với FCM khởi tạo U ngẫu nhiên, PSO có lúc tốt hơn, như các bài báo ghi nhận. Nhưng k-means++ đạt cùng nghiệm mà nhanh hơn.",
    "Các chỉ số PC, AUC, Silhouette cũng không phân biệt được các cách khởi tạo."], { coThan: 16.5 });
  ghiChu(sl, `Đây là kết quả chính của giai đoạn 1. Trên 2 bộ chính với 3 cụm, phiên bản nào, ô lưới nào cũng đạt Jm tốt nhất ở
    cả 30 seed, y như FCM khởi tạo k-means++. Có điều PSO chậm hơn, từ 1,7 lần trên Việt Nam tới 23,5 lần trên Taiwan
    Default. Ở cấu hình có 2 cực tiểu, tỉ lệ đạt dao động từ 77 đến 100 phần trăm, và không theo phiên bản nào. So
    với FCM khởi tạo ngẫu nhiên thì PSO có lúc thắng, đúng như các bài báo viết, nhưng đó là một đối chứng yếu.`);

  sl = trang(pres, 3, "TN2: vì sao? Hàm mục tiêu hầu như chỉ có 1 cực tiểu");
  soLon(sl, 0.6, 1.6, 4.2, "{soLieuESauSoCauHinhMotCucTieu}/{soLieuESauSoCauHinh}", "cấu hình (bộ dữ liệu × m × c) chỉ có 1 mức Jm cuối quan sát được");
  gach(sl, ["2 bộ chính, m = 1,5 và 2, c = 2…8: FCM từ 120 điểm xuất phát (4 kiểu khởi tạo); gom Jm cuối lệch ≤ 0,0001 thành một mức.",
    "Một mức nghĩa là mọi điểm xuất phát rơi về cùng một nghiệm: PSO và FCM đang giảm cùng một hàm.",
    "Ở 2 cấu hình còn lại, k-means++ đã gần như luôn trúng; chạy lại k-means++ vài lần là đủ.",
    "Đây là số cực tiểu quan sát được, không phải chứng minh hàm chỉ có từng ấy cực tiểu."],
    { x: 5.1, y: 1.6, w: 7.65, h: 3.6 }, 15.5);
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.45, w: 12.15, h: 1.15, fill: { color: "FCEFD9" },
    line: { color: "FCEFD9" }, rectRadius: 0.1 });
  sl.addText("Kết luận giai đoạn 1: trên dữ liệu tín dụng đã thử, PSO tìm tâm không giúp FCM; nên chẩn đoán số cực tiểu trước khi nghĩ tới PSO cho tâm.",
    { x: 0.85, y: 5.5, w: 11.7, h: 1.05, fontFace: BF, fontSize: 16, bold: true, color: C.navy, margin: 0, valign: "middle" });
  ghiChu(sl, `TN2 trả lời câu hỏi vì sao. Với mỗi bộ dữ liệu, mỗi giá trị m và c, nhóm chạy FCM từ 120 điểm xuất phát, 4 kiểu khởi tạo, rồi
    đếm xem có bao nhiêu mức Jm cuối. Có 26 trên 28 cấu hình chỉ ra đúng 1 mức, tức là xuất phát từ đâu cũng về cùng
    một nghiệm. Lúc đó PSO và FCM đang cùng giảm một hàm chỉ có 1 đáy, nên khởi tạo tốt hơn cũng không đổi được kết
    quả. Em lưu ý đây chỉ là số cực tiểu quan sát được, chưa phải phép chứng minh. Kết luận của giai đoạn 1 là nên
    chẩn đoán số cực tiểu trước khi dùng PSO cho tâm. Từ đây nhóm chuyển hướng.`);
}

// ---------------------------------------------------------------- Phần 3 (tiếp): Giang, chuyển hướng
function phan3(pres) {
  let sl = trang(pres, 3, "Vì sao chuyển hướng sang trọng số thuộc tính");
  the(pres, sl, 0.6, 1.55, 5.95, 4.95, "Tài liệu nói gì", [
    "Các bài PSO–FCM báo tốt hơn FCM chủ yếu khi so với FCM khởi tạo ngẫu nhiên và không cùng ngân sách tính toán.",
    "Các so sánh trong tài liệu (Mehdizadeh, Ghamisi, Omran, Izakian) cho thấy PSO–FCM hoà hoặc không hơn, và chậm hơn nhiều.",
    "Kết quả TN1, TN2 của nhóm khớp với các so sánh này."], { coThan: 16.5 });
  the(pres, sl, 6.8, 1.55, 5.95, 4.95, "Câu hỏi đặt lại", [
    "PSO nên tối ưu thứ FCM không tự tối ưu được.",
    "Trọng số thuộc tính là thứ đó: FCM coi mọi thuộc tính ngang nhau.",
    "Bài 06 (Liu 2021) đã đánh trọng số trong FCM bằng công thức; O'Neill 2018 dùng PSO tìm trọng số cho phân cụm cứng.",
    "Nhóm đưa PSO tìm trọng số vào FCM trên dữ liệu tín dụng."], { mau: "DDEFE6", coThan: 16.5 });
  ghiChu(sl, `Xong giai đoạn 1, nhóm đọc lại tài liệu và thấy các bài PSO–FCM báo tốt hơn FCM chủ yếu vì đối chứng là
    FCM khởi tạo ngẫu nhiên, mà PSO lại được tính hàm nhiều lần hơn. Còn những so sánh cùng điều kiện trong tài liệu
    thì cho thấy PSO–FCM hoà hoặc không hơn, lại chậm hơn, khớp với TN1 và TN2 của nhóm. Nên nhóm đặt lại câu hỏi: PSO
    nên tối ưu cái mà FCM không tự tối ưu được, ở đây là trọng số thuộc tính. Bài 06 đã đánh trọng số trong FCM bằng
    công thức, còn O'Neill 2018 dùng PSO tìm trọng số cho phân cụm cứng. Nhóm đưa cách làm của O'Neill vào FCM, chính là đề xuất Hải đã trình bày. Bây giờ em đi vào thí nghiệm kiểm tra đề xuất đó.`);

}

// ---------------------------------------------------------------- Phần 2 (tiếp): Hải, đề xuất PSO tìm trọng số
function phanDeXuat(pres) {
  let sl = trang(pres, 2, "Đề xuất: PSO tìm trọng số thuộc tính cho FCM");
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.5, w: 6.3, h: 2.25, fill: { color: C.pale }, line: { color: C.pale },
    rectRadius: 0.1 });
  anh(sl, CT(6), 0.8, 1.6, { w: 5.9, h: 0.95 });
  anh(sl, CT(7), 0.8, 2.65, { w: 5.9, h: 0.95 });
  the(pres, sl, 7.15, 1.5, 5.6, 2.25, "Một hạt = một vector trọng số λ", ["Mỗi thuộc tính một trọng số trong [0,05; 1].",
    "Cận dưới 0,05 giữ mọi thuộc tính: đánh trọng số, không loại bỏ."], { coThan: 16 });
  const b3 = [["1", "FCM với khoảng cách có trọng số", "từ một bộ tâm k-means++ cố định"],
    ["2", "Silhouette trên không gian gốc", "của phân hoạch cứng, mẫu cố định 1.000 điểm"],
    ["3", "Phạt cụm quá nhỏ", "khi cụm nhỏ nhất dưới 5% khách hàng"]];
  b3.forEach(([so, a, b], i) => {
    const x = 0.6 + i * 4.15;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.05, w: 3.85, h: 1.6, fill: { color: C.tint }, line: { color: C.tint },
      rectRadius: 0.1 });
    soTron(pres, sl, x + 0.2, y0(4.05) + 0.2, 0.5, so);
    sl.addText([{ text: a, options: { bold: true, color: C.navy, fontSize: 15, breakLine: true } },
      { text: b, options: { fontSize: 13.5, color: C.ink } }], { x: x + 0.85, y: 4.15, w: 2.85, h: 1.4, fontFace: BF,
      margin: 0, valign: "middle" });
  });
  sl.addText("3 lựa chọn thiết kế của nhóm: Silhouette tính trên không gian gốc, phạt cụm dưới 5%, hạt đầu là λ = 1 (FCM thường). Nhãn rủi ro không đi vào fitness.",
    { x: 0.6, y: 5.85, w: 12.15, h: 0.85, fontFace: BF, fontSize: 15, bold: true, color: C.amber, margin: 0, valign: "middle" });
  ghiChu(sl, `Ở giai đoạn 1, PSO tìm tâm không cải thiện được FCM, Giang sẽ trình bày lý do bằng số liệu. Vì vậy ở giai đoạn 2 nhóm dùng PSO cho trọng số thuộc tính. Đây là đề xuất của nhóm. Mỗi hạt là một vector trọng số lambda, mỗi thuộc tính có một trọng số từ 0,05 tới 1, và
    khoảng cách giữa khách hàng với tâm cụm được tính có trọng số. Fitness của một hạt tính qua 3 bước. Bước một,
    chạy FCM với khoảng cách có trọng số, xuất phát từ cùng một bộ tâm k-means++. Bước hai, lấy phân hoạch cứng, tức gán mỗi khách hàng vào cụm có độ thuộc cao nhất, rồi tính Silhouette trên không gian gốc, chưa nhân trọng số, để PSO không thắng được bằng cách bóp méo không gian.
    Bước ba, cộng phạt nếu có cụm dưới 5 phần trăm khách hàng. Hạt đầu tiên là lambda bằng 1, tức FCM thường, nên về
    fitness PSO không bao giờ kém điểm xuất phát. Nhãn rủi ro không dùng trong fitness, chỉ dùng để đánh giá. Em mời Giang trình bày phần thực nghiệm.`);

}

// ---------------------------------------------------------------- Phần 3 (tiếp): Giang, TN3
function phanTN3(pres) {
  let sl = trang(pres, 3, "TN3: thiết kế so sánh");
  const hang = [["Đối chứng", "Trả lời câu hỏi"],
    ["FCM thường (λ = 1)", "Đánh trọng số có giúp phân biệt rủi ro rõ hơn không?"],
    ["WFCM, trọng số tính bằng công thức", "Có cần PSO không, hay công thức là đủ?"],
    ["Tìm ngẫu nhiên cùng ngân sách", "Lợi ích có do cơ chế bầy đàn, hay chỉ do đánh trọng số?"],
    ["K-means không trọng số (mốc tham chiếu)", "So với một cách phân cụm cứng đơn giản thì sao?"]];
  sl.addTable(hang.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i % 2 ? C.pale : C.white) } } }))),
    { x: 0.6, y: 1.55, w: 12.15, colW: [5.0, 7.15], fontFace: BF, fontSize: 15, rowH: 0.62, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, margin: [4, 10, 4, 10] });
  [["7 bộ × c = 2, 3, 4", "{soLieuETamSoCauHinh} cấu hình, 30 seed"], ["PSO-W, 500 lần", "tính fitness; tìm ngẫu nhiên cùng 500 lần"],
   ["Wilcoxon + Holm", "theo cặp seed, Holm trên {soLieuETamSoPhep} phép so"]].forEach(([a, b], i) => {
    const x = 0.6 + i * 4.15;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.0, w: 3.85, h: 1.55, fill: { color: C.tint }, line: { color: C.tint },
      rectRadius: 0.1 });
    sl.addText([{ text: s(a), options: { bold: true, fontSize: 18, color: C.navy, fontFace: HF, breakLine: true } },
      { text: s(b), options: { fontSize: 13.5, color: C.ink } }], { x: x + 0.2, y: 5.1, w: 3.45, h: 1.35, fontFace: BF,
      margin: 0, valign: "middle" });
  });
  ghiChu(sl, `TN3 so PSO tìm trọng số với 3 đối chứng, mỗi đối chứng trả lời một câu hỏi riêng. FCM thường cho biết đánh trọng số
    có giúp không. WFCM, tức trọng số tính bằng công thức, cho biết có cần tới PSO không. Còn tìm ngẫu nhiên với cùng
    500 lần tính fitness thì cho biết lợi ích có phải nhờ cơ chế bầy đàn không. Sau đợt rà soát, nhóm thêm K-means không trọng số làm mốc tham chiếu; K-means là phân cụm cứng, mỗi điểm thuộc đúng một cụm. Thí nghiệm chạy trên 7 bộ, 3 số cụm, mỗi cấu hình 30 seed. Các phép so dùng
    Wilcoxon theo cặp seed, hiệu chỉnh Holm trên 189 phép so.`);

  sl = trang(pres, 3, "TN3: PSO giúp các cụm phân biệt rủi ro rõ hơn FCM thường");
  const nhanC = ["c = 2", "c = 3", "c = 4"], cs = ["Hai", "Ba", "Bon"];
  const so = (k) => parseFloat(n(k).replace(",", "."));
  sl.addChart(pres.charts.BAR, [
    { name: "FCM thường", labels: nhanC, values: cs.map((c) => so(`soLieuETamVietNamC${c}FcmKppChenhTiLe`)) },
    { name: "FCM + trọng số PSO", labels: nhanC, values: cs.map((c) => so(`soLieuETamVietNamC${c}PsoTrongSoChenhTiLe`)) }],
    { x: 0.6, y: 1.5, w: 6.6, h: 4.9, barDir: "col", barGrouping: "clustered", chartColors: ["9AA9C2", C.navy],
      showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelPosition: "outEnd",
      showLegend: true, ...FONT_BD, legendPos: "b", legendFontSize: 12, catAxisLabelFontSize: 13, valAxisLabelFontSize: 11,
      valAxisMinVal: 0, valAxisMaxVal: 0.7, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
      showTitle: true, title: "Việt Nam: chênh tỉ lệ khó khăn tài chính giữa các cụm (Δr)", titleFontSize: 13,
      titleColor: C.navy });
  soLon(sl, 7.6, 1.5, 5.1, "{soLieuETamVietNamCHaiFcmKppChenhTiLe} → {soLieuETamVietNamCHaiPsoTrongSoChenhTiLe}", "Δr trên Việt Nam, 2 cụm; tốt hơn có ý nghĩa ở mọi c");
  soLon(sl, 7.6, 3.3, 5.1, "{soLieuETamTaiwanKiemChungCBaFcmKppAucCung} → {soLieuETamTaiwanKiemChungCBaPsoTrongSoAucCung}", "AUC phân hoạch trên Taiwan Default, 3 cụm; tốt hơn ở c = 3, 4", C.green);
  sl.addText(s("Trên Taiwan Default c = 3, trọng số tính bằng công thức (WFCM) gần như mất khả năng phân biệt rủi ro: AUC {soLieuETamTaiwanKiemChungCBaWfcmGiaiTichAucCung}."),
    { x: 7.6, y: 5.15, w: 5.1, h: 1.25, fontFace: BF, fontSize: 14, color: C.ink, margin: 0, valign: "top" });
  ghiChu(sl, `Đây là kết quả chính. Trên bộ Việt Nam, ở cả 3 số cụm, trọng số do PSO tìm làm chênh tỉ lệ khó khăn tài chính giữa
    các cụm tăng rõ. Với 2 cụm, mức chênh tăng từ 0,284 lên 0,438. Trên Taiwan Default với 3 và 4 cụm thì AUC của
    phân hoạch tăng, ví dụ với 3 cụm là từ 0,601 lên 0,664. Còn WFCM, trọng số tính bằng công thức, lại gần như không phân biệt được rủi ro trên Taiwan Default, AUC chỉ 0,510. Lý do là công thức dồn trọng số vào các cột số tiền còn nợ trên sao kê vì chúng
    làm cụm chặt nhất, chứ không phải các cột phân biệt được rủi ro. Tiếp theo em phân tích kỹ hơn kết quả này.`);

}

// ---------------------------------------------------------------- Phần 3 (tiếp): Giang, kết quả và diễn giải
function phanKetQua(pres) {
  const cs = ["Hai", "Ba", "Bon"];
  let sl = trang(pres, 3, "Cơ chế bầy đàn: PSO so với tìm ngẫu nhiên cùng ngân sách");
  anh(sl, ANH("e8-hoi-tu.png"), 0.6, 1.5, { w: 8.2, h: 3.6 });
  sl.addText("Fitness tốt nhất theo số lần tính fitness; đường là trung vị, dải là IQR của 30 seed.",
    { x: 0.6, y: 5.15, w: 8.2, h: 0.4, fontFace: BF, fontSize: 11.5, italic: true, color: C.muted, margin: 0 });
  soLon(sl, 9.2, 1.5, 3.5, "{soLieuETamSilhouetteNgauNhienTrongSoHon}/{soLieuETamSoCauHinh}", "cấu hình PSO tối ưu fitness tốt hơn tìm ngẫu nhiên");
  soLon(sl, 9.2, 3.3, 3.5, "{soLieuETamChenhTiLeNgauNhienTrongSoHon}/{soLieuETamSoCauHinh}", "cấu hình PSO phân biệt rủi ro (Δr) tốt hơn, chủ yếu trên Việt Nam", C.amber);
  sl.addText("Bầy đàn giúp tối ưu fitness rõ; về phân biệt rủi ro, lợi ích chỉ rõ trên Việt Nam, bộ khác tìm ngẫu nhiên đủ nhiều cũng đạt.",
    { x: 0.6, y: 5.75, w: 12.15, h: 0.85, fontFace: BF, fontSize: 15, color: C.ink, margin: 0 });
  ghiChu(sl, `Vế thứ hai của CH2 là lợi ích có phải nhờ cơ chế bầy đàn không. Trên hình, đường PSO và đường tìm ngẫu nhiên tách nhau
    sau khoảng 50 tới 100 lần tính fitness. PSO cho fitness tốt hơn có ý nghĩa ở 12 trên 21 cấu hình. Về phân biệt rủi ro thì khác: theo Δr, PSO chỉ hơn tìm ngẫu nhiên ở 4 cấu hình, chủ yếu trên Việt Nam. Ở các bộ khác, lợi ích chủ yếu
    đến từ việc đánh trọng số theo Silhouette, tìm ngẫu nhiên đủ nhiều cũng đạt được.`);

  sl = trang(pres, 3, "Mốc tham chiếu K-means: lợi ích đo so với FCM m = 2");
  const kh = [["Bộ, số cụm", "FCM thường", "K-means", "FCM + trọng số PSO"]];
  [["VietNam", "Việt Nam"], ["TaiwanKiemChung", "Taiwan Default"]].forEach(([b, ten]) => cs.forEach((c, i) => kh.push(
    [`${ten}, c = ${i + 2}`, n(`soLieuETam${b}C${c}FcmKppChenhTiLe`), n(`soLieuETamKm${b}C${c}ChenhTiLe`),
      n(`soLieuETam${b}C${c}PsoTrongSoChenhTiLe`)])));
  sl.addTable(kh.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, align: j ? "center" : "left",
    color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.navy : (i % 2 ? C.pale : C.white) } } }))),
    { x: 0.6, y: 1.5, w: 7.2, colW: [2.4, 1.5, 1.5, 1.8], fontFace: BF, fontSize: 14, rowH: 0.52, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, margin: [3, 8, 3, 8] });
  sl.addText("Chênh tỉ lệ vỡ nợ / khó khăn tài chính Δr, trung vị 30 seed, cùng seed và tiền xử lý.",
    { x: 0.6, y: 5.25, w: 7.2, h: 0.4, fontFace: BF, fontSize: 11.5, italic: true, color: C.muted, margin: 0 });
  the(pres, sl, 8.1, 1.5, 4.65, 5.05, "Đọc bảng", ["2 cụm: PSO tốt hơn K-means trên cả 2 bộ chính.",
    "Ba, 4 cụm: K-means tốt hơn có ý nghĩa ở Việt Nam và Taiwan c = 4 (Δr); FCM m = 2 gần suy biến ở đây.",
    "Lợi ích của PSO là so với FCM m = 2. Giữ FCM khi cần độ thuộc để nhận ra khách hàng biên."], { mau: "FCEFD9", coThan: 16 });
  ghiChu(sl, `Trong lúc rà soát, nhóm thêm K-means không trọng số làm mốc tham chiếu, dùng cùng seed và cùng tiền xử lý. Với 2
    cụm, PSO tốt hơn K-means trên cả 2 bộ chính. Với 3 và 4 cụm thì K-means phân biệt rủi ro tốt hơn có ý nghĩa trên Việt Nam và trên Taiwan Default với
    4 cụm; Taiwan 3 cụm thì không phát hiện khác biệt. Lý do là ở các cấu hình này FCM với m bằng 2 gần suy biến, tức 2 tâm gần trùng nhau nên cụm mất ý nghĩa. Cho nên lợi ích của PSO tìm trọng số mà nhóm báo
    cáo là lợi ích so với FCM m bằng 2. Nếu chỉ cần phân hoạch cứng thì phải so với K-means trước. FCM đáng dùng khi
    cần độ thuộc, em sẽ nói ở slide phân khúc.`);

  sl = trang(pres, 3, "Trọng số tìm được có nghĩa kinh tế");
  anh(sl, ANH("e8-trong-so.png"), 0.6, 1.45, { w: 8.0, h: 4.6 });
  sl.addText("Trọng số trung vị 30 seed (c = 3), chuẩn hoá theo trung bình; đường 1,0 là FCM thường.",
    { x: 0.6, y: 6.1, w: 8.0, h: 0.4, fontFace: BF, fontSize: 11.5, italic: true, color: C.muted, margin: 0 });
  the(pres, sl, 8.9, 1.45, 3.85, 2.45, "Taiwan Default", ["{soLieuETamTaiwanKiemChungTrongSoCaoMotTen} {soLieuETamTaiwanKiemChungTrongSoCaoMotGiaTri} · {soLieuETamTaiwanKiemChungTrongSoCaoHaiTen} {soLieuETamTaiwanKiemChungTrongSoCaoHaiGiaTri} · {soLieuETamTaiwanKiemChungTrongSoCaoBaTen} {soLieuETamTaiwanKiemChungTrongSoCaoBaGiaTri}",
    "Tình trạng trả nợ các tháng gần đây (PAY_x: tháng x trễ hạn hay không)."], { coThan: 15.5 });
  the(pres, sl, 8.9, 4.1, 3.85, 2.45, "Việt Nam", ["{soLieuETamVietNamTrongSoCaoMotTen} (giá trị công ty / doanh thu) · {soLieuETamVietNamTrongSoCaoHaiTen} (tổng nợ / vốn chủ sở hữu) · {soLieuETamVietNamTrongSoCaoBaTen} (giá cổ phiếu / doanh thu)",
    "Giá trị công ty so với doanh thu, và mức vay nợ."], { coThan: 14, mau: "DDEFE6" });
  ghiChu(sl, `Tiếp theo là trọng số mà PSO tìm được. Trên Taiwan Default, 3 thuộc tính nặng nhất là tình trạng trả nợ các
    tháng gần đây, PAY_3, PAY_2 và PAY_0. Đây cũng là thứ ngân hàng nhìn vào khi xét thẻ tín dụng. Các cột dư nợ thì
    bị giảm trọng số. Trên Việt Nam, PSO tăng trọng số cho giá trị cả công ty so với doanh thu, tổng nợ so với vốn chủ sở hữu và giá cổ phiếu so
    với doanh thu, tức là nhóm định giá công ty và mức vay nợ. Nhãn không đi vào fitness mà PSO vẫn tự chọn ra các thuộc tính này, nên nhóm coi đây
    là dấu hiệu tốt. Giờ em so sánh 5 phiên bản PSO.`);
}
function y0(v) { return v; }

// ---------------------------------------------------------------- Phần 3 (tiếp): Giang, TN4 và đánh giá
function phan4(pres) {
  let sl = trang(pres, 3, "TN4: 5 phiên bản PSO khi tìm trọng số");
  const ten = [["Goc", "PSO gốc"], ["W", "PSO-W"], ["S", "PSO-S"], ["Log", "PSO-LOG"], ["Qpso", "QPSO"]];
  const bang = [["Phiên bản", "Hơn tìm ngẫu nhiên: fitness", "Hơn tìm ngẫu nhiên: Δr", "Khác PSO gốc"]];
  ten.forEach(([k, t]) => bang.push([t, `${n(`soLieuEChin${k}NgauNhienDiemHon`)}/${n("soLieuEChinSoO")} ô`,
    `${n(`soLieuEChin${k}NgauNhienChenhTiLeHon`)}/${n("soLieuEChinSoO")} ô`, k === "Goc" ? "—" : "không ô nào"]));
  sl.addTable(bang.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, align: j ? "center" : "left",
    color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.navy : (i % 2 ? C.pale : C.white) } } }))),
    { x: 0.6, y: 1.5, w: 8.0, colW: [1.7, 2.3, 2.2, 1.8], fontFace: BF, fontSize: 14.5, rowH: 0.6, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, margin: [3, 8, 3, 8] });
  sl.addText(s("12 ô = 2 bộ chính × c = 2, 3, 4 × 2 ngân sách (500 và 1.000 lần tính fitness); Holm trên {soLieuEChinSoPhep} phép so."),
    { x: 0.6, y: 5.25, w: 8.0, h: 0.6, fontFace: BF, fontSize: 12, italic: true, color: C.muted, margin: 0 });
  the(pres, sl, 8.9, 1.5, 3.85, 5.1, "Đọc bảng", ["Mọi phiên bản đều hơn tìm ngẫu nhiên.",
    "Không phát hiện bản cải tiến nào khác PSO gốc sau hiệu chỉnh Holm.",
    "Khác vai trò tìm tâm, ở đây thứ hạng tối ưu không còn."], { mau: "FCEFD9", coThan: 16 });
  ghiChu(sl, `TN4 lắp cả 5 phiên bản vào cùng một khung tìm trọng số, chạy trên 2 bộ chính, 3 số cụm và 2 mức ngân sách, tổng
    cộng 12 ô. Phiên bản nào cũng hơn tìm ngẫu nhiên: về fitness ở 11 tới 12 ô, về Δr ở 8 tới 9 ô. Nhưng sau hiệu
    chỉnh Holm, nhóm không phát hiện bản cải tiến nào khác PSO gốc ở ô nào. Họ phép so lớn nên kiểm định kém nhạy.
    Điều nhóm khẳng định được là nếu có khác biệt thì khác biệt đó nhỏ.`);

  sl = trang(pres, 3, "Ngân sách tính toán: 500 hay 1.000 lần?");
  anh(sl, ANH("e9-quy-dao.png"), 0.6, 1.5, { w: 8.2, h: 3.7 });
  sl.addText("Fitness tốt nhất theo số lần tính fitness, 5 phiên bản, trung vị 30 seed.",
    { x: 0.6, y: 5.25, w: 8.2, h: 0.4, fontFace: BF, fontSize: 11.5, italic: true, color: C.muted, margin: 0 });
  soLon(sl, 9.2, 1.5, 3.5, "27/30", "trường hợp fitness tốt lên khi tăng 500 → 1.000 lần");
  soLon(sl, 9.2, 3.3, 3.5, "× 2", "thời gian, trong khi kết luận về Δr và AUC không đổi", C.amber);
  sl.addText("Khuyến nghị: PSO gốc hoặc PSO-W với khoảng 500 lần tính fitness là đủ.",
    { x: 0.6, y: 5.85, w: 12.15, h: 0.75, fontFace: BF, fontSize: 16, bold: true, color: C.navy, margin: 0 });
  ghiChu(sl, `Khi tăng ngân sách từ 500 lên 1.000 lần tính fitness, fitness tốt lên ở 27 trên 30 trường hợp, rõ nhất khi có nhiều
    cụm. Có điều thời gian gấp đôi, còn kết luận về 2 chỉ số chính thì không đổi. Trên hình, các đường gần như chồng
    lên nhau, đến cuối mới tách ra. Nên nhóm khuyến nghị dùng PSO gốc hoặc PSO-W với khoảng 500 lần tính fitness.`);

  sl = trang(pres, 3, "Diễn giải phân khúc: Việt Nam, 3 cụm");
  const nhom = ["Cụm rủi ro thấp", "Cụm giữa", "Cụm rủi ro cao"], ma = ["Mot", "Hai", "Ba"];
  sl.addChart(pres.charts.BAR, [
    { name: "FCM thường", labels: nhom, values: ma.map((m) => so4(`soLieuEBonVietNamFcmKppCum${m}VoNo`)) },
    { name: "FCM + trọng số PSO", labels: nhom, values: ma.map((m) => so4(`soLieuEBonVietNamPsoTrongSoCum${m}VoNo`)) }],
    { x: 0.6, y: 1.5, w: 6.4, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: ["9AA9C2", C.navy], showValue: true,
      dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelPosition: "outEnd", showLegend: true, ...FONT_BD, legendPos: "b",
      legendFontSize: 12, catAxisLabelFontSize: 12.5, valAxisLabelFontSize: 11, valAxisMinVal: 0, valAxisMaxVal: 60,
      valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true,
      title: "Tỉ lệ khó khăn tài chính trong cụm (%)", titleFontSize: 13, titleColor: C.navy });
  soLon(sl, 7.4, 1.5, 5.3, "{soLieuEBonVietNamFcmKppCumBaCo}% → {soLieuEBonVietNamPsoTrongSoCumBaCo}%", "doanh nghiệp trong cụm rủi ro cao: cụm gọn hơn, tỉ lệ {soLieuEBonVietNamFcmKppCumBaVoNo}% → {soLieuEBonVietNamPsoTrongSoCumBaVoNo}%");
  soLon(sl, 7.4, 3.3, 5.3, "{soLieuEBonVietNamFcmKppCumBaMoHo}–{soLieuEBonVietNamFcmKppCumMotMoHo}% → {soLieuEBonVietNamPsoTrongSoCumMotMoHo}–{soLieuEBonVietNamPsoTrongSoCumBaMoHo}%", "doanh nghiệp \"mơ hồ\": độ thuộc có ý nghĩa trở lại", C.green, 22);
  sl.addText("Cụm rủi ro cao có nợ so với vốn chủ sở hữu (X6, X8, X10) cao nhất; cụm giữa tách theo giá cổ phiếu so với doanh thu (P/S).",
    { x: 7.4, y: 5.25, w: 5.3, h: 1.25, fontFace: BF, fontSize: 14, color: C.ink, margin: 0, valign: "top" });
  ghiChu(sl, `Bây giờ em diễn giải các phân khúc trên bộ Việt Nam với 3 cụm. Với FCM thường, cụm rủi ro cao chiếm 34,3 phần trăm
    doanh nghiệp, tỉ lệ khó khăn tài chính là 35,4 phần trăm. Khi dùng trọng số do PSO tìm, cụm này chỉ còn 19,6 phần
    trăm doanh nghiệp nhưng tỉ lệ lên 49,8 phần trăm, tức là cụm gọn và rõ hơn. Doanh nghiệp trong cụm rủi ro cao có
    các tỉ số nợ trên vốn chủ sở hữu cao nhất. Với phân cụm mờ, điểm quan trọng là tỉ lệ doanh nghiệp "mơ hồ" giảm từ
    90 tới 100 phần trăm xuống khoảng 18 tới 27 phần trăm, nên độ thuộc lại phân biệt được doanh nghiệp điển hình với
    doanh nghiệp biên.`);

  sl = trang(pres, 3, "Đánh giá chung các thuật toán");
  const y = 1.55;
  [["Phiên bản PSO có thứ hạng rõ khi đo như thuật toán tối ưu", "PSO-W và QPSO đưa gbest sát cực tiểu hơn PSO gốc, đúng lý do chúng được đề xuất."],
   ["Thứ hạng đó không chuyển thành kết quả phân cụm", "Tìm tâm: FCM đưa mọi điểm xuất phát về cùng nghiệm. Tìm trọng số: PSO gốc đã đủ."],
   ["Vai trò của PSO quyết định, không phải phiên bản", "Tìm tâm không đem lại gì; tìm trọng số giúp các cụm phân biệt rủi ro tốt hơn FCM thường."]]
    .forEach(([a, b], i) => {
      soTron(pres, sl, 0.6, y + i * 1.25, 0.6, i + 1);
      sl.addText([{ text: a, options: { bold: true, color: C.navy, fontSize: 17, breakLine: true } },
        { text: b, options: { fontSize: 14.5, color: C.ink } }], { x: 1.4, y: y - 0.08 + i * 1.25, w: 11.3, h: 1.15,
        fontFace: BF, margin: 0, valign: "top" });
    });
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.4, w: 12.15, h: 1.25, fill: { color: "FCEFD9" },
    line: { color: "FCEFD9" }, rectRadius: 0.1 });
  sl.addText("Khuyến nghị: cần độ thuộc thì dùng FCM k-means++ + PSO gốc/PSO-W tìm trọng số (khoảng 500 lần), giữ WFCM làm phương án rẻ để so; chỉ cần phân hoạch cứng với từ 3 cụm thì so với K-means trước.",
    { x: 0.85, y: 5.45, w: 11.7, h: 1.15, fontFace: BF, fontSize: 15, bold: true, color: C.navy, margin: 0, valign: "middle" });
  ghiChu(sl, `Xét cả 2 vai trò, nhóm có 3 nhận xét. Thứ nhất, nếu đo như một thuật toán tối ưu thì các bản cải tiến xếp hạng rõ
    ràng, đúng như tài liệu viết. Thứ hai, thứ hạng đó không chuyển thành kết quả phân cụm. Thứ ba, cái quyết định là
    PSO được giao vai trò gì: vẫn 5 thuật toán đó, dùng tìm tâm thì không đem lại gì, dùng tìm trọng số thì có ích.
    Về thực hành, khi cần độ thuộc thì dùng FCM khởi tạo k-means++ cộng PSO gốc hoặc PSO-W tìm trọng số. Khi chỉ cần
    phân hoạch cứng với từ 3 cụm thì so với K-means trước. Tiếp theo là phần kiểm chứng.`);
}
function so4(k) { return parseFloat(n(k).replace(",", ".")); }

// ---------------------------------------------------------------- Phần 3 (tiếp): Giang, kiểm chứng và kết luận
function phan5(pres) {
  let sl = trang(pres, 3, "TN5: kiểm chứng trên mẫu độc lập");
  const kh = [["Mẫu nguồn", "PSO học trọng số"], ["Áp trọng số", "không học lại"], ["Mẫu đích", "so với FCM thường và học tại chỗ"]];
  kh.forEach(([a, b], i) => {
    const x = 0.6 + i * 4.15;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 3.6, h: 1.45, fill: { color: i === 1 ? "FCEFD9" : C.tint },
      line: { color: C.tint }, rectRadius: 0.1 });
    sl.addText([{ text: a, options: { bold: true, fontSize: 18, color: C.navy, fontFace: HF, breakLine: true } },
      { text: b, options: { fontSize: 14, color: C.ink } }], { x: x + 0.2, y: 1.6, w: 3.2, h: 1.35, fontFace: BF, margin: 0,
      align: "center", valign: "middle" });
    if (i < 2) muiTen(pres, sl, x + 3.65, 2.07, 0.45);
  });
  soLon(sl, 0.6, 3.35, 3.9, "{soLieuEMuoiChenhTiLeFcmKppHon}/{soLieuEMuoiSoCauHinh}", "cấu hình trọng số chuyển sang tốt hơn FCM thường (Δr), không cấu hình nào kém");
  soLon(sl, 4.75, 3.35, 3.9, "{soLieuEMuoiChenhTiLePsoTaiChoBang}/{soLieuEMuoiSoCauHinh}", "cấu hình không khác học ngay trên mẫu đích (Δr)", C.green);
  soLon(sl, 8.9, 3.35, 3.85, "{soLieuEMuoiTaiwanTkKcCBonPsoChuyenAucCung} vs {soLieuEMuoiTaiwanTkKcCBonPsoTaiChoAucCung}", "AUC: chuyển sang so với học tại chỗ, Taiwan thiết kế → kiểm chứng, c = 4", C.amber, 30);
  sl.addText("2 chiều chuyển trên mỗi bộ chính: Taiwan Default (mẫu thiết kế ↔ mẫu kiểm chứng), Việt Nam (2 nửa chia theo doanh nghiệp).",
    { x: 0.6, y: 5.75, w: 12.15, h: 0.8, fontFace: BF, fontSize: 14.5, color: C.ink, margin: 0 });
  ghiChu(sl, `Thí nghiệm cuối là TN5. TN5 hỏi trọng số học trên một mẫu có dùng được cho mẫu khác không. Nhóm học trọng
    số trên mẫu nguồn, áp nguyên sang mẫu đích, rồi so với FCM thường và với PSO học ngay trên mẫu đích. Nhóm làm trên
    2 bộ chính, mỗi bộ 2 chiều chuyển, 3 số cụm. Theo Δr, trọng số chuyển sang tốt hơn FCM thường ở 10 trên 12
    cấu hình, không cấu hình nào kém hơn. So với học tại chỗ thì không khác ở 11 trên 12 cấu hình. Có mất mát ở một
    chiều của Taiwan Default: với 4 cụm, AUC là 0,572, so với 0,645 khi học tại chỗ, nhưng vẫn cao hơn FCM thường.`);

  sl = trang(pres, 3, "Khi nào PSO tìm trọng số không có ích, và giới hạn");
  the(pres, sl, 0.6, 1.5, 5.95, 5.1, "Không có ích khi", [
    "2 cụm: kém FCM thường theo AUC ở {soLieuETamAucCungFcmKppKem} cấu hình, kể cả Taiwan Default (chênh nhỏ), vì fitness không nhìn nhãn.",
    "Bộ mà FCM thường đã gần sụp (South German, Australian, Taiwan Bankruptcy): trọng số không cứu được.",
    "Có ích khi dữ liệu có nhiều thuộc tính số liên tục đo các khía cạnh khác nhau của rủi ro."], { coThan: 16 });
  the(pres, sl, 6.8, 1.5, 5.95, 5.1, "Giới hạn của nghiên cứu", [
    "Đối chứng chính là FCM m = 2; K-means phân biệt rủi ro tốt hơn khi có từ 3 cụm.",
    "Kiểm định lấy seed làm đơn vị: đo dao động của thuật toán, chưa đo dao động do lấy mẫu khách hàng.",
    "Nhãn bộ Việt Nam định nghĩa từ chính các đại lượng tài chính và gán theo công ty.",
    "Chỉ 2 bộ chính; một fitness, m = 2."], { coThan: 16, mau: "FCEFD9" });
  ghiChu(sl, `Đề xuất này cũng có lúc không có ích. Với 2 cụm, PSO kém FCM thường theo AUC ở 5 cấu hình, vì fitness không nhìn
    nhãn: tách đẹp về hình học chưa chắc là phân biệt đúng rủi ro. Trên các bộ mà FCM thường đã gần sụp thì trọng số
    cũng không cứu được. Nghiên cứu còn mấy giới hạn. Đối chứng chính là FCM m bằng 2, trong khi K-means tốt hơn khi
    có từ 3 cụm. Kiểm định lấy seed làm đơn vị nên chưa đo được dao động do lấy mẫu khách hàng. Nhãn của bộ Việt Nam
    được định nghĩa từ chính các chỉ số tài chính. Và nhóm mới có 2 bộ dữ liệu chính.`);

  sl = trang(pres, 3, "Quy trình đề xuất 3 bước");
  [["Chẩn đoán số cực tiểu", "Chạy FCM từ nhiều điểm xuất phát; nếu chỉ 1 nghiệm thì khởi tạo k-means++ là đủ, không cần PSO cho tâm."],
   ["PSO tìm trọng số thuộc tính", "PSO gốc hoặc PSO-W, khoảng 500 lần tính fitness; Silhouette trên không gian gốc, phạt cụm nhỏ."],
   ["Kiểm bằng nhãn nếu có", "Có nhãn: chỉ áp trọng số khi phân khúc tốt hơn trên mẫu không dùng để tìm trọng số. Chưa có nhãn: dùng song song FCM thường, kiểm lại khi có kết cục thực tế."]]
    .forEach(([a, b], i) => {
      const x = 0.6 + i * 4.15;
      sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 3.85, h: 4.3, fill: { color: [C.tint, "DDEFE6", "FCEFD9"][i] },
        line: { color: C.tint }, rectRadius: 0.12 });
      soTron(pres, sl, x + 0.3, 1.9, 0.75, i + 1);
      sl.addText([{ text: a, options: { bold: true, fontSize: 20, color: C.navy, fontFace: HF, breakLine: true } },
        { text: " ", options: { fontSize: 6, breakLine: true } },
        { text: b, options: { fontSize: 15, color: C.ink } }], { x: x + 0.3, y: 2.85, w: 3.3, h: 2.9, fontFace: BF, margin: 0,
        valign: "top" });
      if (i < 2) muiTen(pres, sl, x + 3.88, 3.55, 0.25);
    });
  sl.addText("Bước 3 dùng nhãn (nếu có) để chọn có áp trọng số hay không, như TN5 đã làm. Bước 1 và 2 không cần nhãn.",
    { x: 0.6, y: 6.1, w: 12.15, h: 0.5, fontFace: BF, fontSize: 14, italic: true, color: C.muted, margin: 0 });
  ghiChu(sl, `Từ các kết quả này, nhóm đề xuất một quy trình 3 bước cho người làm phân khúc tín dụng bằng FCM. Bước 1 là chẩn đoán
    số cực tiểu: chạy FCM từ nhiều điểm xuất phát, nếu chỉ ra 1 nghiệm thì k-means++ là đủ, không cần PSO tìm tâm.
    Bước 2 là dùng PSO gốc hoặc PSO-W tìm trọng số thuộc tính, khoảng 500 lần tính fitness. Bước 3: fitness không dùng
    nhãn, nên nếu có nhãn lịch sử thì chỉ áp trọng số khi nó làm phân khúc tốt hơn trên một mẫu có nhãn không dùng để tìm
    trọng số. Nếu dữ liệu mới chưa có nhãn thì bước 1 và 2 vẫn chạy được, nhưng chưa khẳng định được trọng số có ích,
    nên dùng song song với FCM thường và kiểm lại khi có kết cục thực tế.`);

  sl = trang(pres, 3, "Kết luận");
  const kl = [["CH1", "PSO tìm tâm không giúp FCM: Jm hầu như chỉ có 1 cực tiểu ({soLieuESauSoCauHinhMotCucTieu}/{soLieuESauSoCauHinh} cấu hình), mọi phiên bản cho cùng nghiệm với k-means++ mà chậm hơn."],
    ["CH2", "PSO tìm trọng số giúp các cụm phân biệt rủi ro tốt hơn FCM thường trên 2 bộ chính; lợi ích do bầy đàn rõ trên Việt Nam; K-means vẫn là mốc mạnh khi có từ 3 cụm."],
    ["CH3", "Mọi phiên bản hơn tìm ngẫu nhiên; không phát hiện bản cải tiến nào khác PSO gốc. Vai trò quan trọng hơn phiên bản."],
    ["CH4", "Trọng số giữ lợi ích trên mẫu độc lập ({soLieuEMuoiChenhTiLeFcmKppHon}/{soLieuEMuoiSoCauHinh} cấu hình theo Δr); không có ích với 2 cụm và trên bộ FCM gần sụp."]];
  kl.forEach(([a, b], i) => {
    const yy = 1.5 + i * 1.28;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: yy, w: 12.15, h: 1.12, fill: { color: i % 2 ? C.white : C.pale },
      line: { color: C.line }, rectRadius: 0.08 });
    sl.addText(a, { x: 0.8, y: yy, w: 1.0, h: 1.12, fontFace: HF, fontSize: 22, bold: true, color: C.amber, margin: 0,
      valign: "middle" });
    sl.addText(s(b), { x: 1.85, y: yy + 0.06, w: 10.7, h: 1.0, fontFace: BF, fontSize: 15, color: C.ink, margin: 0,
      valign: "middle" });
  });
  ghiChu(sl, `Em trả lời lại 4 câu hỏi. CH1: trên dữ liệu tín dụng nhóm đã thử, PSO tìm tâm không giúp được FCM, vì hàm mục tiêu
    hầu như chỉ có 1 cực tiểu. CH2: PSO tìm trọng số giúp các cụm phân biệt rủi ro tốt hơn FCM thường trên 2 bộ chính.
    Lợi ích do cơ chế bầy đàn thấy rõ trên Việt Nam, còn K-means vẫn là mốc mạnh khi có từ 3 cụm. CH3: phiên bản nào
    cũng hơn tìm ngẫu nhiên, không phát hiện bản cải tiến nào khác PSO gốc. CH4: lợi ích vẫn giữ được trên mẫu độc lập. Nói gọn
    lại, giao cho PSO vai trò gì mới là điều quyết định, chọn phiên bản PSO nào thì không.`);

  sl = pres.addSlide();
  sl.background = { color: C.navy };
  sl.addText("Em xin cảm ơn Thầy và các bạn đã lắng nghe", { x: 0.7, y: 1.4, w: 11.9, h: 1.3, fontFace: HF, fontSize: 38,
    bold: true, color: C.white, margin: 0 });
  sl.addText("Phần 3 · Thực nghiệm, kết luận · Trần Trường Giang", { x: 0.7, y: 2.75, w: 11.9, h: 0.6, fontFace: BF,
    fontSize: 20, italic: true, color: "F2C48D", margin: 0 });
  ghiChu(sl, `Phần trình bày của nhóm em đến đây là hết. Em xin cảm ơn Thầy và các bạn đã lắng nghe.`);
}

// ---------------------------------------------------------------- xuất tệp
function tao(tenTep, cacPhan) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "Nhóm 1";
  pres.title = "Ứng dụng PSO kết hợp Fuzzy C-means trong đánh giá rủi ro tín dụng";
  cacPhan.forEach((f) => f(pres));
  return pres.write({ outputType: "nodebuffer" }).then(sauXuLy).then((buf) => {
    fs.writeFileSync(path.join(GOC, tenTep), buf);
    console.log("Đã ghi", tenTep);
  });
}

// Be Vietnam Pro thiếu chữ Hy Lạp và mũi tên: tách các ký tự đó thành run riêng với font có sẵn trên Windows/Mac.
const DU_PHONG = [[/[Ͱ-Ͽ←-⇿]/u, "Arial"]];
const fontDuPhong = (c) => (DU_PHONG.find(([re]) => re.test(c)) || [null, null])[1];

function tachRun(_, rPr, text) {
  const doan = [];
  for (const c of text) {
    const f = fontDuPhong(c);
    if (doan.length && doan[doan.length - 1].f === f) doan[doan.length - 1].t += c;
    else doan.push({ f, t: c });
  }
  return doan.map(({ f, t }) => {
    const r = f ? rPr.replace(/(<a:(?:latin|ea|cs) typeface=")[^"]*"/g, `$1${f}"`) : rPr;
    return `<a:r>${r}<a:t>${t}</a:t></a:r>`;
  }).join("");
}

async function sauXuLy(buf) {
  const zip = await JSZip.loadAsync(buf);
  const tep = Object.keys(zip.files).filter((f) => /^ppt\/(slides|charts|notesSlides)\/[^/]+\.xml$/.test(f));
  for (const f of tep) {
    let x = await zip.file(f).async("string");
    x = x.replace(/ lang="en-US"/g, ' lang="vi-VN"');
    x = x.replace(/<a:r>(<a:rPr[^>]*\/>|<a:rPr[^>]*>[\s\S]*?<\/a:rPr>)<a:t>([^<]*)<\/a:t><\/a:r>/g, tachRun);
    zip.file(f, x);
  }
  return zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
}

(async () => {
  const ha = [phan1], hai = [phanNen, phan2, phanDeXuat], giang = [phanTN12, phan3, phanTN3, phanKetQua, phan4, phan5];
  await tao("phan-1-ha-ly-thuyet.pptx", ha);
  await tao("phan-2-hai-thuat-toan-va-mo-hinh.pptx", hai);
  await tao("phan-3-giang-thuc-nghiem.pptx", giang);
  await tao("tong-hop-nhom-1-pso-fcm.pptx", [...ha, ...hai, ...giang]);
})().catch((e) => { console.error(e); process.exit(1); });
