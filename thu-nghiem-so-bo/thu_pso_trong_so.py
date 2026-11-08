# -*- coding: utf-8 -*-
"""Thử nhanh: PSO (gốc, cải tiến) tìm trọng số thuộc tính cho FCM có vượt đối chứng không, sau khi đã chọn m tốt.

    python3 thu_pso_trong_so.py [--so-seed 10] [--tien-trinh 16] [--ra kq.csv]

Hạt là w ∈ [0,05; 1]^D; FCM phân cụm trên X·√(w / mean w) từ bộ tâm k-means++ cố định của lượt (J_m chỉ có một cực
tiểu nên khởi tạo không ảnh hưởng). Hai fitness:
    sil  −Silhouette trên không gian gốc (mẫu cố định 1000 điểm) — không dùng nhãn
    auc  −AUC tách rủi ro trên nửa HUẤN LUYỆN (cụm rủi ro chọn theo nhãn huấn luyện) — có dùng nhãn
cộng phạt khi cụm cứng nhỏ nhất < 5%. Mọi phương pháp báo AUC trên nửa KIỂM TRA (chia phân tầng 50/50 theo seed).
Phiên bản tìm kiếm: goc (ω = 1), w (ω 0,9 → 0,4), qpso, ngau_nhien (cùng số lần chấm, không học).
Đối chứng: fcm (w đều), wfcm (trọng số tính bằng công thức, cach_ket_hop.wfcm_giai_tich).
"""

from __future__ import annotations

import argparse
import os
import sys

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from multiprocessing import Pool  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score  # noqa: E402

from cach_ket_hop import phat_kich_thuoc, wfcm_giai_tich  # noqa: E402
from chay_thuc_nghiem import nap  # noqa: E402
from fcm import FCM, cdist_euclidean  # noqa: E402
from pso_co_so import pso  # noqa: E402
from pso_khoi_tao import C_GOC, OMEGA  # noqa: E402
from thuat_toan_b import qpso  # noqa: E402

CAU_HINH = [(bo, m, c) for bo in ("viet_nam", "taiwan_kiem_chung") for m in (1.5, 2.0) for c in (3, 5)]
PHIEN_BAN = ("goc", "w", "qpso", "ngau_nhien")
FITNESS = ("sil", "auc")
N_HAT, SO_VONG = 10, 30
W_MIN = 0.05
VONG_FCM_TRONG = 200
N_MAU_SIL = 1000


def chia_phan_tang(y, rng):
    tr = np.concatenate([rng.permutation(np.flatnonzero(y == k))[: (y == k).sum() // 2] for k in (0, 1)])
    return np.sort(tr), np.setdiff1d(np.arange(len(y)), tr)


def auc_tach(U, y, tr, te):
    nhan = U.argmax(1)
    ti_le = [y[tr][nhan[tr] == k].mean() if (nhan[tr] == k).any() else -1.0 for k in range(U.shape[1])]
    k = int(np.argmax(ti_le))
    return roc_auc_score(y[tr], U[tr, k]), roc_auc_score(y[te], U[te, k])


def he_so(w):
    return np.sqrt(w / w.mean())


class BaiToan:
    def __init__(self, X, y, c, m, rng):
        self.X, self.y, self.c, self.m = X, y, c, m
        self.tr, self.te = chia_phan_tang(y, rng)
        self.V0 = FCM(c, m=m).init_centroids_kmeans_plusplus(X, seed=rng)
        self.mau = rng.choice(len(X), min(len(X), N_MAU_SIL), replace=False)
        self.D_mau = cdist_euclidean(X[self.mau], X[self.mau])
        np.fill_diagonal(self.D_mau, 0.0)

    def phan_cum(self, w, max_iter=VONG_FCM_TRONG):
        s = he_so(w)
        U, _, _ = FCM(self.c, m=self.m, max_iter=max_iter).fit(self.X * s, init_v=self.V0 * s, with_u=False)
        return U

    def sil(self, U):
        nhan = U[self.mau].argmax(1)
        return silhouette_score(self.D_mau, nhan, metric="precomputed") if len(np.unique(nhan)) > 1 else -1.0

    def fitness(self, loai):
        def f(w):
            U = self.phan_cum(w)
            diem = self.sil(U) if loai == "sil" else auc_tach(U, self.y, self.tr, self.te)[0]
            return -diem + phat_kich_thuoc(U)
        return f

    def do(self, U):
        auc_tr, auc_te = auc_tach(U, self.y, self.tr, self.te)
        return dict(auc_te=auc_te, auc_tr=auc_tr, sil=self.sil(U),
                    cum_nho=np.bincount(U.argmax(1), minlength=self.c).min() / len(U))


def tim(bt, loai, pb, rng):
    d = bt.X.shape[1]
    bay = np.vstack([np.ones(d), rng.uniform(W_MIN, 1.0, size=(N_HAT - 1, d))])
    f = bt.fitness(loai)
    if pb == "ngau_nhien":
        ung_vien = np.vstack([bay, rng.uniform(W_MIN, 1.0, size=(N_HAT * SO_VONG, d))])
        diem = np.array([f(w) for w in ung_vien])
        return ung_vien[int(np.argmin(diem))]
    if pb == "qpso":
        return qpso(bay, W_MIN, 1.0, f, rng, max_vong=SO_VONG, dung_tri_tre=False)[0]
    return pso(bay, W_MIN, 1.0, f, rng, omega=OMEGA[pb], max_vong=SO_VONG, c1=C_GOC, c2=C_GOC,
               dung_tri_tre=False).gbest


def viec(tham_so):
    bo, m, c, seed = tham_so
    du_lieu = nap(bo)
    X, y = du_lieu.X, du_lieu.y
    bt = BaiToan(X, y, c, m, np.random.default_rng(seed))
    khoa = dict(bo=bo, m=m, c=c, seed=seed)
    dong = []
    t0 = perf_counter()
    dong.append({**khoa, "pp": "fcm", **bt.do(bt.phan_cum(np.ones(X.shape[1]), max_iter=10000)), "t": perf_counter() - t0})
    t0 = perf_counter()
    kq = wfcm_giai_tich(X, c, m, np.random.default_rng(seed))
    dong.append({**khoa, "pp": "wfcm", **bt.do(kq.U), "t": perf_counter() - t0})
    for loai in FITNESS:
        for pb in PHIEN_BAN:
            t0 = perf_counter()
            w = tim(bt, loai, pb, np.random.default_rng(seed + 1))
            U = bt.phan_cum(w, max_iter=10000)
            top = ",".join(du_lieu.ten_cot[i] for i in np.argsort(-w)[:3])
            dong.append({**khoa, "pp": f"{loai}_{pb}", **bt.do(U), "t": perf_counter() - t0, "top3": top})
    return dong


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-seed", type=int, default=10)
    ap.add_argument("--tien-trinh", type=int, default=16)
    ap.add_argument("--ra", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ket-qua-pso-trong-so.csv"))
    a = ap.parse_args(argv)
    viec_lam = [(bo, m, c, 1000 + s) for bo, m, c in CAU_HINH for s in range(a.so_seed)]
    t0 = perf_counter()
    with Pool(a.tien_trinh) as p:
        d = pd.DataFrame([r for ds in p.imap_unordered(viec, viec_lam) for r in ds])
    d.to_csv(a.ra, index=False)
    print(f"{len(viec_lam)} tác vụ, {perf_counter() - t0:.0f} s → {a.ra}")
    g = d.groupby(["bo", "m", "c", "pp"])[["auc_te", "auc_tr", "sil", "cum_nho", "t"]].median().round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
