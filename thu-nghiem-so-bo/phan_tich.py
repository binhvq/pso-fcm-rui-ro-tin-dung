# -*- coding: utf-8 -*-
"""Phân tích kết quả thí nghiệm khởi tạo. Chạy: python3 phan_tich.py ket-qua.csv"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TOL_TARGET = 1e-3  # gbest coi là "tới đích" khi ≤ J*·(1 + 0,1%)
TOL_BEST = 1e-6  # nghiệm cuối coi là "tốt nhất" khi ≤ J*·(1 + 1e-6)
PSO = ["ngau_nhien_deu", "diem_du_lieu", "tent", "kmeanspp", "doi_lap", "de_xuat"]

d = pd.read_csv(sys.argv[1])
d["J_star"] = d.groupby(["dataset", "c"]).J.transform("min")


def fe_to_target(row):
    if not isinstance(row.trace, str) or not row.trace:
        return np.nan
    target = row.J_star * (1 + TOL_TARGET)
    for item in row.trace.split(";"):
        fe, g = item.split(":")
        if float(g) <= target:
            return float(fe)
    return np.nan


d["fe_dich"] = d.apply(fe_to_target, axis=1)
d["toi_dich"] = d.fe_dich.notna()
d["tot_nhat"] = d.J <= d.J_star * (1 + TOL_BEST)
d["fe_dich_phat"] = d.fe_dich.fillna(d.fe)  # chưa tới đích → tính bằng tổng FE đã dùng

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)

print("=== Tỉ lệ lượt FCM/PSO–FCM sụp (tâm trùng gốc) theo bộ dữ liệu")
print(d.groupby("dataset").sup.mean().round(3).to_string(), "\n")

agg = d.groupby(["dataset", "c", "method"]).agg(
    t_tong_ms=("t_total", lambda x: 1000 * x.median()),
    fe_dich=("fe_dich_phat", "median"),
    toi_dich=("toi_dich", "mean"),
    vong_pso=("pso_iter", "median"),
    vong_fcm=("fcm_iter", "median"),
    tot_nhat=("tot_nhat", "mean"),
    J=("J", "median"),
).round(3)
print("=== Trung vị theo (bộ dữ liệu, c, phương pháp)")
print(agg.to_string(), "\n")

print("=== Đề xuất so với ngẫu nhiên đều (Wilcoxon theo cặp seed)")
rows = []
for (ds, c), g in d.groupby(["dataset", "c"]):
    a = g[g.method == "de_xuat"].set_index("seed")
    b = g[g.method == "ngau_nhien_deu"].set_index("seed").loc[a.index]
    k = g[g.method == "kmeanspp"].set_index("seed").loc[a.index]
    out = dict(dataset=ds, c=c, sup=bool(g.sup.mean() > 0.5))
    for col, name in (("fe_dich_phat", "fe_dich"), ("t_total", "t_tong"), ("fcm_iter", "vong_fcm"), ("J", "J")):
        diff = a[col].values - b[col].values
        out[f"{name}_ratio"] = round(float(np.median(a[col]) / np.median(b[col])), 3) if np.median(b[col]) else np.nan
        out[f"{name}_p"] = float(wilcoxon(a[col], b[col]).pvalue) if np.any(diff != 0) else 1.0
    out["fe_dich_vs_kpp_ratio"] = round(float(np.median(a.fe_dich_phat) / np.median(k.fe_dich_phat)), 3)
    out["fe_dich_vs_kpp_p"] = float(wilcoxon(a.fe_dich_phat, k.fe_dich_phat).pvalue) if np.any(a.fe_dich_phat.values != k.fe_dich_phat.values) else 1.0
    rows.append(out)
w = pd.DataFrame(rows)
for col in [c for c in w.columns if c.endswith("_p")]:
    w[col] = w[col].map(lambda p: f"{p:.1e}")
print(w.to_string(index=False), "\n")

print("=== FCM thuần so với PSO–FCM (trung vị, bộ không sụp)")
ok = d[~d.dataset.isin(d.groupby("dataset").sup.mean().loc[lambda s: s > 0.5].index)]
print(ok.groupby(["dataset", "method"]).agg(t_ms=("t_total", lambda x: 1000 * x.median()), tot_nhat=("tot_nhat", "mean")).round(3).to_string())
