# -*- coding: utf-8 -*-
"""Phân tích thí nghiệm PSO–XB có ràng buộc. Chạy: python3 phan_tich_xb_rang_buoc.py ket-qua.csv"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv(sys.argv[1])
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)

cols = ["t", "XB", "PC", "silhouette", "cum_nho_nhat", "chenh_ty_le", "auc_cum_rui_ro"]
print("=== Trung vị theo (bộ dữ liệu, c, phương pháp)")
print(d.groupby(["dataset", "c", "method"])[cols].median().to_string(float_format=lambda x: f"{x:.3g}"), "\n")

print("=== Tỉ lệ lượt có cụm cứng nhỏ nhất ≥ 5% / ≥ 10% dữ liệu")
d["dat5"] = d.cum_nho_nhat >= 0.05
d["dat10"] = d.cum_nho_nhat >= 0.10
print(d.pivot_table(index=["dataset", "c"], columns="method", values="dat5", aggfunc="mean").round(2).to_string())
print(d.pivot_table(index=["dataset", "c"], columns="method", values="dat10", aggfunc="mean").round(2).to_string(), "\n")


def so_sanh(a_name, b_name):
    rows = []
    for (ds, c), g in d.groupby(["dataset", "c"]):
        a = g[g.method == a_name].set_index("seed").sort_index()
        b = g[g.method == b_name].set_index("seed").loc[a.index]
        out = dict(dataset=ds, c=c)
        for col in ("auc_cum_rui_ro", "chenh_ty_le", "silhouette", "t"):
            x, y = a[col].values, b[col].values
            out[f"{col}"] = f"{np.median(x):.3f} / {np.median(y):.3f}"
            out[f"{col}_p"] = f"{wilcoxon(x, y).pvalue:.1e}" if np.any(x != y) else "—"
        rows.append(out)
    print(f"=== {a_name} so với {b_name} (trung vị a / b, Wilcoxon theo cặp seed)")
    print(pd.DataFrame(rows).to_string(index=False), "\n")


so_sanh("pso_xb_rb5", "fcm")
so_sanh("pso_xb_rb5", "fcm_lap_XB")
so_sanh("pso_xb_rb5", "pso_xb")
so_sanh("pso_xb_rb10", "fcm")
