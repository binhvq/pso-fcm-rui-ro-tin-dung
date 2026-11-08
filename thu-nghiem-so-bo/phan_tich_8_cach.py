# -*- coding: utf-8 -*-
"""Phân tích thí nghiệm 8 cách kết hợp PSO–FCM theo tiêu chí đã chốt trước.

Cải thiện = AUC tách rủi ro cao hơn FCM 1 lần (trung vị), Wilcoxon theo cặp seed với p sau hiệu chỉnh Holm < 0,05
(hiệu chỉnh chung cho mọi phép so với FCM), và trung vị cụm cứng nhỏ nhất ≥ 5% dữ liệu.

Chạy: python3 phan_tich_8_cach.py ket-qua.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv(sys.argv[1])
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 500)

cols = ["auc_cum_rui_ro", "chenh_ty_le", "silhouette", "PC", "XB", "cum_nho_nhat", "t"]
print("=== Trung vị theo (bộ dữ liệu, c, phương pháp)")
print(d.groupby(["dataset", "c", "method"])[cols].median().to_string(float_format=lambda x: f"{x:.3g}"), "\n")


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


rows = []
for (ds, c), g in d.groupby(["dataset", "c"]):
    base = g[g.method == "fcm"].set_index("seed").sort_index()
    for m in sorted(set(g.method) - {"fcm"}):
        a = g[g.method == m].set_index("seed").loc[base.index]
        x, y = a.auc_cum_rui_ro.values, base.auc_cum_rui_ro.values
        p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        rows.append(dict(dataset=ds, c=c, method=m, auc=np.median(x), auc_fcm=np.median(y), p=p,
                         cum_nho_nhat=np.median(a.cum_nho_nhat), t=np.median(a.t), t_fcm=np.median(base.t)))
r = pd.DataFrame(rows)
r["p_holm"] = holm(r.p)
r["cai_thien"] = (r.auc > r.auc_fcm) & (r.p_holm < 0.05) & (r.cum_nho_nhat >= 0.05)
r["kem_hon"] = (r.auc < r.auc_fcm) & (r.p_holm < 0.05)

print("=== So với FCM 1 lần (AUC tách rủi ro; Holm trên", len(r), "phép so)")
show = r.copy()
show["p_holm"] = show.p_holm.map(lambda v: f"{v:.1e}")
print(show.drop(columns="p").to_string(index=False, float_format=lambda x: f"{x:.3f}"), "\n")

print("=== Số cấu hình (trên 6) cải thiện / kém hơn có ý nghĩa so với FCM")
print(r.groupby("method")[["cai_thien", "kem_hon"]].sum().astype(int).sort_values("cai_thien", ascending=False).to_string(), "\n")

print("=== Tách riêng đóng góp của PSO: c7_pso_trong_so so với c7_wfcm; c8_pso_kfcm so với c8_kfcm")
for a_name, b_name in (("c7_pso_trong_so", "c7_wfcm"), ("c8_pso_kfcm", "c8_kfcm")):
    for (ds, c), g in d.groupby(["dataset", "c"]):
        a = g[g.method == a_name].set_index("seed").sort_index()
        b = g[g.method == b_name].set_index("seed").loc[a.index]
        x, y = a.auc_cum_rui_ro.values, b.auc_cum_rui_ro.values
        p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        print(f"{a_name} vs {b_name} | {ds} c={c}: AUC {np.median(x):.3f} / {np.median(y):.3f}, p = {p:.1e}, "
              f"cụm nhỏ nhất {np.median(a.cum_nho_nhat):.3f} / {np.median(b.cum_nho_nhat):.3f}")
