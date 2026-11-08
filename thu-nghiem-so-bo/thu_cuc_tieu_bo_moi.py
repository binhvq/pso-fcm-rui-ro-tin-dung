# -*- coding: utf-8 -*-
"""Chẩn đoán số cực tiểu của J_m (m = 2) trên hai bộ tín dụng doanh nghiệp mới: Polish Companies Bankruptcy (UCI 365)
và Taiwanese Bankruptcy Prediction (UCI 572). Cùng cách đếm với thu_cuc_tieu_dia_phuong.py (30 khởi tạo U ngẫu nhiên,
gom J lệch tương đối < 1e-4), để so được với Việt Nam và Taiwan Default.

Tiền xử lý: điền trung vị cho giá trị thiếu, bỏ cột hằng; bản "log" = log có dấu rồi z-score, bản "khong_log" = z-score.
Mẫu con phân tầng 5.000 dòng (rng(0)) nếu bộ lớn hơn.

Chạy: python3 thu_cuc_tieu_bo_moi.py <thư mục data/raw> <file CSV kết quả>
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
from scipy.io import arff
from scipy.stats import kurtosis, skew

from thu_pso_sau import FCM, signed_log, zscore

N_START = 30
M = 2.0
CS = (2, 3, 4, 5, 6, 8)
TOL = 1e-4


def mau_con(X, y, n=5000, seed=0):
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], int(round(n * np.mean(y == k))), replace=False)
                          for k in np.unique(y)])
    return X[idx], y[idx]


def lam_sach(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.nunique(dropna=True) > 1]
    return df.fillna(df.median()).values.astype(float)


def load_variants(raw):
    out, info = {}, {}
    files = sorted(glob.glob(os.path.join(raw, "polish-bankruptcy", "**", "3year.arff"), recursive=True))
    data, _ = arff.loadarff(files[0])
    p = pd.DataFrame(data)
    y = p.pop("class").astype(str).str.strip("b'").astype(int).values
    X, y = mau_con(lam_sach(p), y)
    info["polish_3year"] = (len(y), float(np.mean(y)))
    out["polish_log"], out["polish_khong_log"] = zscore(signed_log(X)), zscore(X)

    files = glob.glob(os.path.join(raw, "taiwan-bankruptcy", "**", "*.csv"), recursive=True)
    t = pd.read_csv(files[0])
    y = t.pop("Bankrupt?").values
    X, y = mau_con(lam_sach(t), y)
    info["taiwan_bankruptcy"] = (len(y), float(np.mean(y)))
    out["tw_phasan_log"], out["tw_phasan_khong_log"] = zscore(signed_log(X)), zscore(X)
    return out, info


DATA, INFO = load_variants(sys.argv[1])


def run_cfg(args):
    name, c = args
    X = DATA[name]
    J, sup = [], 0
    for s in range(N_START):
        model = FCM(c, m=M, max_iter=3000)
        _, V, _ = model.fit(X, seed=s)
        J.append(model.objective_from_centroids(X, V))
        sup += int(np.abs(V).max() < 1e-3)
    J = np.sort(np.array(J))
    minima = [J[0]]
    for j in J[1:]:
        if (j - minima[-1]) > TOL * abs(minima[-1]):
            minima.append(j)
    return dict(dataset=name, c=c, so_cuc_tieu=len(minima), ty_le_tot=float(np.mean(J <= J[0] * (1 + TOL))),
                sup=sup / N_START, chenh_J=(J[-1] - J[0]) / abs(J[0]))


if __name__ == "__main__":
    print("Mẫu, tỉ lệ phá sản:", INFO)
    mo_ta = [dict(dataset=n, n=X.shape[0], D=X.shape[1], lech=float(np.median(np.abs(skew(X)))),
                  nhon=float(np.median(kurtosis(X))), so_gia_tri_rieng=float(np.median([len(np.unique(c)) for c in X.T])))
             for n, X in DATA.items()]
    with Pool(8) as pool:
        rows = pool.map(run_cfg, [(n, c) for n in DATA for c in CS])
    df = pd.DataFrame(rows).merge(pd.DataFrame(mo_ta), on="dataset")
    df.to_csv(sys.argv[2], index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(mo_ta).round(2).to_string(index=False), "\n")
    for key in ("so_cuc_tieu", "ty_le_tot", "sup", "chenh_J"):
        print(f"=== {key}")
        print(df.pivot_table(index="dataset", columns="c", values=key).round(3).to_string(), "\n")
