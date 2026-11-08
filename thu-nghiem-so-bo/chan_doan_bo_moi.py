# -*- coding: utf-8 -*-
"""Chẩn đoán nhanh một bộ dữ liệu tín dụng mới: J_m của FCM có nhiều cực tiểu không, và PSO khởi tạo có giúp không.

    python3 chan_doan_bo_moi.py <csv> --nhan <cột nhãn> [--duong <giá trị nhãn dương>] [--c 2 3 4 5 6]
        [--mau 5000] [--so-luot 30] [--khong-log] [--bo-cot a b] [--sep ,] [--am-la-thieu] [--ra kq.csv]

Chỉ dùng các cột số (cột định danh bị bỏ, in ra danh sách). Tiền xử lý như các bộ chính: điền trung vị, bỏ cột hằng,
log có dấu (trừ khi --khong-log), z-score; bộ lớn hơn --mau dòng lấy mẫu phân tầng theo nhãn (seed 0).
Mỗi c: FCM từ 4 kiểu khởi tạo (U ngẫu nhiên, c điểm ngẫu nhiên, k-means++, PSO-W 20 hạt 30 vòng), mỗi kiểu --so-luot lượt.
"""

from __future__ import annotations

import argparse
import os
import sys

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import tien_xu_ly as tx  # noqa: E402
from fcm import FCM  # noqa: E402
from giao_dien import fcm_kpp, fcm_mot_lan  # noqa: E402
from pso_khoi_tao import fcm_diem_ngau_nhien, ham_e5  # noqa: E402

TOL_J, NGUONG_SUP = 1e-4, 1e-3
KIEU = {"u_ngau_nhien": fcm_mot_lan, "diem_ngau_nhien": fcm_diem_ngau_nhien, "kmeans_pp": fcm_kpp,
        "pso_w": ham_e5("w", 20, 30)}


def nap_csv(duong_dan, nhan, duong, sep, bo_cot, mau, log, am_la_thieu=False):
    df = pd.read_csv(duong_dan, sep=sep, na_values=["?", "NA", ""])
    y_tho = df.pop(nhan)
    y = (y_tho.astype(str) == str(duong)).to_numpy().astype(int) if duong is not None else y_tho.to_numpy().astype(int)
    df = df.drop(columns=[c for c in bo_cot if c in df], errors="ignore")
    so = df.apply(pd.to_numeric, errors="coerce")
    giu = so.notna().mean() >= 0.9
    print(f"Bỏ {int((~giu).sum())} cột không phải số: {list(so.columns[~giu])[:15]}")
    so = so.loc[:, giu]
    if am_la_thieu:
        so = so.mask(so < 0)
    so = so.loc[:, so.nunique(dropna=True) > 1]
    so = so.fillna(so.median())
    X, y = so.to_numpy(float), y
    if len(X) > mau:
        chon = tx.mau_phan_tang(y, mau, np.random.default_rng(0))
        X, y = X[chon], y[chon]
    X = tx.signed_log(X) if log else X
    X, ten = tx.zscore(X, list(so.columns))
    return X, y, ten


def auc_rui_ro(U, y):
    nhan = U.argmax(1)
    ti_le = [y[nhan == k].mean() if (nhan == k).any() else -1 for k in range(U.shape[1])]
    return roc_auc_score(y, U[:, int(np.argmax(ti_le))]) if 0 < y.sum() < len(y) else np.nan


def chan_doan(X, y, cs, so_luot):
    dong = []
    for c in cs:
        mo_hinh = FCM(c, m=2.0)
        for kieu, ham in KIEU.items():
            for s in range(so_luot):
                kq = ham(X, c, 2.0, np.random.default_rng(1000 + s), None)
                dong.append(dict(c=c, kieu=kieu, seed=1000 + s, J=mo_hinh.objective_from_centroids(X, kq.V),
                                 sup=float(np.abs(kq.V).max() < NGUONG_SUP), cum_nho=float(np.bincount(
                                     kq.U.argmax(1), minlength=c).min() / len(X)), pc=float((kq.U ** 2).sum() / len(X)),
                                 auc=auc_rui_ro(kq.U, y), t=kq.t_tong))
    return pd.DataFrame(dong)


def so_muc(J):
    J = np.sort(np.asarray(J))
    return int(1 + np.sum(np.diff(J) > TOL_J * np.abs(J[:-1])))


def tom_tat(d):
    ra = []
    for c, dc in d.groupby("c"):
        j_tot = dc["J"].min()
        r = dict(c=c, so_muc_J=so_muc(dc["J"]))
        for kieu, dk in dc.groupby("kieu", sort=False):
            r[f"{kieu}_dat"] = round(100 * (dk["J"] <= (1 + TOL_J) * j_tot).mean())
        r["sup_%"] = round(100 * dc["sup"].mean())
        r["pc_tv"] = round(dc["pc"].median(), 3)
        r["cum_nho_tv_%"] = round(100 * dc["cum_nho"].median(), 1)
        r["auc_tv"] = round(dc["auc"].median(), 3)
        ra.append(r)
    return pd.DataFrame(ra)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--nhan", required=True)
    ap.add_argument("--duong")
    ap.add_argument("--c", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--mau", type=int, default=5000)
    ap.add_argument("--so-luot", type=int, default=30)
    ap.add_argument("--khong-log", action="store_true")
    ap.add_argument("--bo-cot", nargs="*", default=[])
    ap.add_argument("--sep", default=",")
    ap.add_argument("--ra")
    ap.add_argument("--am-la-thieu", action="store_true", help="giá trị âm là mã thiếu (FICO HELOC: -7, -8, -9)")
    a = ap.parse_args(argv)
    X, y, ten = nap_csv(a.csv, a.nhan, a.duong, a.sep, a.bo_cot, a.mau, not a.khong_log, a.am_la_thieu)
    print(f"{os.path.basename(a.csv)}: {X.shape[0]} dòng × {X.shape[1]} thuộc tính số; nhãn dương {100 * y.mean():.1f}%")
    d = chan_doan(X, y, a.c, a.so_luot)
    if a.ra:
        d.to_csv(a.ra, index=False)
    print(tom_tat(d).to_string(index=False))


if __name__ == "__main__":
    main()
