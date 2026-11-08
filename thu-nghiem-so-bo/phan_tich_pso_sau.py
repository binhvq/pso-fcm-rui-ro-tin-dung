# -*- coding: utf-8 -*-
"""Phân tích thí nghiệm PSO sâu. Chạy: python3 phan_tich_pso_sau.py ket-qua.csv"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv(sys.argv[1])
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)

cols = ["t", "XB", "PC", "PE", "silhouette", "so_cum_rong", "cum_nho_nhat", "chenh_ty_le", "auc_cum_rui_ro"]
print("=== Trung vị theo (bộ dữ liệu, c, phương pháp)")
print(d.groupby(["dataset", "c", "method"])[cols].median().to_string(float_format=lambda x: f"{x:.3g}"), "\n")

print("=== Tỉ lệ lượt có cụm rỗng (số cụm có điểm < c)")
d["co_cum_rong"] = d.so_cum_rong < d.c
print(d.pivot_table(index=["dataset", "c"], columns="method", values="co_cum_rong", aggfunc="mean").round(2).to_string(), "\n")

print("=== PSO–XB so với FCM thuần (Wilcoxon theo cặp seed)")
rows = []
for (ds, c), g in d.groupby(["dataset", "c"]):
    a = g[g.method == "pso_xb"].set_index("seed").sort_index()
    b = g[g.method == "fcm"].set_index("seed").loc[a.index]
    out = dict(dataset=ds, c=c)
    for col in ("auc_cum_rui_ro", "chenh_ty_le", "silhouette", "PC", "t"):
        x, y = a[col].values, b[col].values
        out[f"{col}_xb"] = round(float(np.median(x)), 3)
        out[f"{col}_fcm"] = round(float(np.median(y)), 3)
        out[f"{col}_p"] = f"{wilcoxon(x, y).pvalue:.1e}" if np.any(x != y) else "—"
    rows.append(out)
print(pd.DataFrame(rows).to_string(index=False))
