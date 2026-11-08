// Xuất mọi macro \newcommand{\soLieu...}{...} trong results/so-lieu*.tex ra so-lieu.json cho tao-slide.js.
//   node xuat-so-lieu.js
"use strict";
const fs = require("fs");
const path = require("path");

const KET_QUA = path.join(__dirname, "..", "results");
const TEP = ["so-lieu", "so-lieu-e5", "so-lieu-e6", "so-lieu-e7", "so-lieu-e8", "so-lieu-e10", "so-lieu-e9", "so-lieu-e8-kmeans"];
const MACRO = /^\\newcommand\{\\(soLieu\w+)\}\{(.*)\}$/;

const soLieu = {};
for (const ten of TEP) {
  const dong = fs.readFileSync(path.join(KET_QUA, `${ten}.tex`), "utf8").split(/\r?\n/);
  dong.forEach((d, i) => {
    if (!d.trim() || d.startsWith("%")) return;
    const m = d.match(MACRO);
    if (!m) throw new Error(`${ten}.tex:${i + 1}: dòng không phải macro số liệu`);
    if (m[1] in soLieu) throw new Error(`${ten}.tex:${i + 1}: macro ${m[1]} trùng`);
    soLieu[m[1]] = m[2];
  });
}
const json = "{\n" + Object.entries(soLieu).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(",\n") + "\n}";
fs.writeFileSync(path.join(__dirname, "so-lieu.json"), json);
console.log(`Đã ghi so-lieu.json: ${Object.keys(soLieu).length} macro`);
