# -*- coding: utf-8 -*-
"""Kiểm tra đa khởi đầu: bộ dữ liệu/cấu hình nào làm FCM kẹt cực tiểu địa phương?

Với mỗi (bộ dữ liệu, tiền xử lý, m, c): chạy FCM từ 30 ma trận U khởi tạo ngẫu nhiên, gom các J cuối
trùng nhau (sai lệch tương đối < 1e-4) thành một cực tiểu, rồi báo:
  so_cuc_tieu   số cực tiểu khác nhau tìm thấy
  ty_le_tot     tỉ lệ lượt đạt J nhỏ nhất
  sup           tỉ lệ lượt sụp (mọi tâm trùng nhau)
kèm đặc trưng mô tả dữ liệu: n, D, trung vị |độ lệch|, trung vị độ nhọn dư.

Chạy: python3 thu_cuc_tieu_dia_phuong.py <file CSV>
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import glob
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from thu_pso_sau import FCM, RAW, signed_log, zscore

N_START = 30
MS = (1.2, 1.5, 2.0)
CS = (2, 3, 4, 5, 6, 8)
TOL = 1e-4


def load_variants():
    out = {}
    v = pd.read_csv(glob.glob(RAW + "/vn-financial-distress-listed-firms/*.csv")[0], encoding="latin-1")
    f = v[[f"X{i}" for i in range(1, 26)]].copy()
    for col in ("X6", "X7", "X12"):
        f[col] = f[col].replace(0, np.nan).fillna(f[col].replace(0, np.nan).median())
    f = f.drop(columns=["X15", "X18", "X21", "X3"]).values.astype(float)
    out["viet_nam_log"] = zscore(signed_log(f))
    out["viet_nam_khong_log"] = zscore(f)

    t = pd.read_excel(glob.glob(RAW + "/taiwan-default/*.xls")[0], header=1)
    y = t.iloc[:, -1].values
    X = t.drop(columns=[t.columns[0], t.columns[-1]]).values.astype(float)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
    out["taiwan_log"] = zscore(signed_log(X[idx]))
    out["taiwan_khong_log"] = zscore(X[idx])

    s = pd.read_csv(RAW + "/south-german-credit/SouthGermanCredit.asc", sep=r"\s+")
    out["south_german"] = zscore(s.drop(columns="kredit").values.astype(float))
    out["german_so"] = zscore(np.loadtxt(RAW + "/german-credit/german.data-numeric")[:, :-1])
    a = np.loadtxt(RAW + "/australian-credit/australian.dat")[:, :-1]
    out["australian_log"] = zscore(signed_log(a))
    out["australian_khong_log"] = zscore(a)
    return out


DATA = load_variants()


def run_cfg(args):
    name, m, c = args
    X = DATA[name]
    J, sup = [], 0
    for s in range(N_START):
        model = FCM(c, m=m, max_iter=3000)
        U, V, _ = model.fit(X, seed=s)
        J.append(model.objective_from_centroids(X, V))
        sup += int(np.abs(V).max() < 1e-3)
    J = np.sort(np.array(J))
    minima = [J[0]]
    for j in J[1:]:
        if (j - minima[-1]) > TOL * abs(minima[-1]):
            minima.append(j)
    return dict(dataset=name, m=m, c=c, so_cuc_tieu=len(minima),
                ty_le_tot=float(np.mean(J <= J[0] * (1 + TOL))), sup=sup / N_START,
                J_tot=J[0], J_te=J[-1], chenh_J=(J[-1] - J[0]) / abs(J[0]))


if __name__ == "__main__":
    mo_ta = []
    for name, X in DATA.items():
        mo_ta.append(dict(dataset=name, n=X.shape[0], D=X.shape[1],
                          lech=float(np.median(np.abs(skew(X)))), nhon=float(np.median(kurtosis(X))),
                          so_gia_tri_rieng=float(np.median([len(np.unique(col)) for col in X.T]))))
    jobs = [(n, m, c) for n in DATA for m in MS for c in CS]
    with Pool(8) as pool:
        rows = pool.map(run_cfg, jobs)
    df = pd.DataFrame(rows).merge(pd.DataFrame(mo_ta), on="dataset")
    df.to_csv(sys.argv[1], index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(mo_ta).round(2).to_string(index=False), "\n")
    for key in ("so_cuc_tieu", "ty_le_tot", "sup"):
        print(f"=== {key}")
        print(df.pivot_table(index=["dataset", "m"], columns="c", values=key).round(2).to_string(), "\n")
