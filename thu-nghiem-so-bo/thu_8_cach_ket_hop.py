# -*- coding: utf-8 -*-
"""Thí nghiệm: thử 8 cách kết hợp PSO với FCM trong tài liệu, so với FCM thuần.

Tiêu chí "cải thiện" chốt TRƯỚC khi chạy: AUC tách rủi ro (nhãn vỡ nợ, không thuật toán nào dùng nhãn) cao hơn FCM
1 lần với p < 0,05 sau hiệu chỉnh Holm, và cụm cứng nhỏ nhất (trung vị) ≥ 5% dữ liệu. Thời gian báo cáo kèm.

Dữ liệu: taiwan_moi — mẫu phân tầng 5.000 khách hàng KHÔNG trùng mẫu đã dùng ở các lần thử trước (kiểm chứng thật);
viet_nam — toàn bộ 3.277 dòng (đã dùng khi thiết kế). m = 2, c ∈ {2, 3, 4}.

Phương pháp:
  fcm, fcm_lap (20 lần chạy lại, chọn J nhỏ nhất), kmeans                      đối chứng
  c1_tuan_tu      PSO tối ưu J_m → FCM tinh chỉnh                              (Zhang 2025, Mane 2014)
  c2_fcm_gieo     nghiệm FCM làm 1 hạt; PSO J_m → FCM tinh chỉnh               (Mehdizadeh 2008)
  c3_xen_ke       mỗi 10 vòng PSO, mỗi hạt chạy 5 bước FCM; cuối FCM tinh chỉnh (Zhang & Ma 2020)
  c4_pso_tam      PSO tối ưu J_m, hạt là tâm, không FCM tinh chỉnh            (Qian 2022, Omran 2005)
  c5_pso_U        PSO tối ưu J_m, hạt là ma trận độ thuộc U                     (Izakian 2009 FPSO, Sun 2024)
  c6_pso_xb       PSO tối ưu Xie–Beni + ràng buộc cụm ≥ 5%                      (Omran ch.6: chỉ số hợp lệ làm fitness)
  c7_pso_trong_so PSO tìm trọng số thuộc tính; FCM có trọng số phân cụm; chấm XB + ràng buộc trên không gian gốc
  c7_wfcm         đối chứng của c7: FCM có trọng số tính giải tích              (kiểu Liu 2021)
  c8_pso_kfcm     PSO trên hàm mục tiêu kernel FCM → kernel FCM tinh chỉnh      (Shang 2021)
  c8_kfcm         đối chứng của c8: kernel FCM thuần

Chạy: python3 thu_8_cach_ket_hop.py <file CSV> [số seed]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from thu_pso_sau import FCM, RAW, cdist_sq_euclidean, danh_gia, signed_log, xb_from_centroids, zscore

M = 2.0
CLUSTERS = (2, 3, 4)
N_P, MAX_ITER = 20, 100
STALL_TOL, STALL_ITERS = 1e-5, 10
W0, W1, C1 = 0.9, 0.4, 1.49445
NGUONG = 0.05
N_RESTART = 20
METHODS = ("fcm", "fcm_lap", "kmeans", "c1_tuan_tu", "c2_fcm_gieo", "c3_xen_ke", "c4_pso_tam", "c5_pso_U",
           "c6_pso_xb", "c7_pso_trong_so", "c7_wfcm", "c8_pso_kfcm", "c8_kfcm")


# ------------------------------------------------------------------ dữ liệu
def load():
    out = {}
    t = pd.read_excel(glob.glob(RAW + "/taiwan-default/*.xls")[0], header=1)
    y = t.iloc[:, -1].values
    X = t.drop(columns=[t.columns[0], t.columns[-1]]).values.astype(float)
    rng = np.random.default_rng(0)  # tái tạo đúng mẫu cũ để loại ra
    cu = np.concatenate([rng.choice(np.where(y == k)[0], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
    con_lai = np.setdiff1d(np.arange(len(y)), cu)
    rng = np.random.default_rng(1)
    moi = np.concatenate([rng.choice(con_lai[y[con_lai] == k], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
    assert len(np.intersect1d(moi, cu)) == 0
    out["taiwan_moi"] = (zscore(signed_log(X[moi])), y[moi])

    v = pd.read_csv(glob.glob(RAW + "/vn-financial-distress-listed-firms/*.csv")[0], encoding="latin-1")
    f = v[[f"X{i}" for i in range(1, 26)]].copy()
    for col in ("X6", "X7", "X12"):
        f[col] = f[col].replace(0, np.nan)
        f[col] = f[col].fillna(f[col].median())
    f = f.drop(columns=["X15", "X18", "X21", "X3"])
    out["viet_nam"] = (zscore(signed_log(f.values.astype(float))), v["Financial_Distress"].values)
    return out


DATA = load()


# ------------------------------------------------------------------ tiện ích
def xb_UV(X, U, V):
    """Xie–Beni của phân hoạch (U, V) trong không gian X."""
    sep = cdist_sq_euclidean(V, V)
    np.fill_diagonal(sep, np.inf)
    return float(np.sum((U**M) * cdist_sq_euclidean(X, V)) / (len(X) * max(sep.min(), 1e-12)))


def phat_kich_thuoc(U, c):
    nho = np.bincount(U.argmax(1), minlength=c).min() / len(U)
    return max(0.0, NGUONG - nho) / NGUONG


def tam_tu_U(X, U):
    W = U**M
    return (W.T @ X) / np.fmax(W.sum(0)[:, None], 1e-12)


def pso_box(init, lb, ub, fitness, rng, hook=None, hook_every=10):
    """PSO hộp giới hạn cho vector/ma trận bất kỳ. init: (N, ...). Trả về (gbest, gbest_f, số FE, số vòng)."""
    P = init.copy()
    vmax = 0.2 * (ub - lb)
    f = np.array([fitness(p) for p in P])
    fe = len(P)
    Vel = np.zeros_like(P)
    pb, pbf = P.copy(), f.copy()
    g = int(np.argmin(f))
    gb, gbf = P[g].copy(), f[g]
    stall, it = 0, 0
    for it in range(1, MAX_ITER + 1):
        w = W0 - (W0 - W1) * (it - 1) / (MAX_ITER - 1)
        Vel = np.clip(w * Vel + C1 * rng.random(P.shape) * (pb - P) + C1 * rng.random(P.shape) * (gb - P), -vmax, vmax)
        P = np.clip(P + Vel, lb, ub)
        if hook is not None and it % hook_every == 0:
            P = np.stack([hook(p) for p in P])
        f = np.array([fitness(p) for p in P])
        fe += len(P)
        better = f < pbf
        pb[better], pbf[better] = P[better], f[better]
        g = int(np.argmin(pbf))
        prev = gbf
        if pbf[g] < gbf:
            gb, gbf = pb[g].copy(), pbf[g]
        stall = stall + 1 if (prev - gbf) <= STALL_TOL * abs(prev) else 0
        if stall >= STALL_ITERS:
            break
    return gb, gbf, fe, it


def init_tam(X, c, rng, model, n=N_P):
    half = n // 2
    kpp = [model.init_centroids_kmeans_plusplus(X, seed=rng) for _ in range(half)]
    pts = [X[rng.choice(len(X), c, replace=False)] for _ in range(n - half)]
    return np.stack(kpp + pts)


# ------------------------------------------------------------------ FCM có trọng số và kernel FCM
def wfcm(X, c, rng, beta=2.0, max_iter=300, tol=1e-5):
    """FCM có trọng số thuộc tính tính giải tích: w_j ∝ (1/D_j)^(1/(β−1)), Σw = 1."""
    n, d = X.shape
    w = np.full(d, 1.0 / d)
    V = FCM(c, m=M).init_centroids_kmeans_plusplus(X, seed=rng)
    for _ in range(max_iter):
        Xs, Vs = X * w**(beta / 2), V * w**(beta / 2)
        U = FCM(c, m=M)._cost_to_membership(cdist_sq_euclidean(Xs, Vs))
        V_new = tam_tu_U(X, U)
        Dj = np.einsum("ik,ikj->j", U**M, (X[:, None, :] - V_new[None, :, :]) ** 2) + 1e-12
        w = (1.0 / Dj) ** (1.0 / (beta - 1))
        w /= w.sum()
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    Xs, Vs = X * w**(beta / 2), V * w**(beta / 2)
    return FCM(c, m=M)._cost_to_membership(cdist_sq_euclidean(Xs, Vs)), V, w


def kfcm_U(X, V, sigma2):
    d = np.fmax(2.0 * (1.0 - np.exp(-cdist_sq_euclidean(X, V) / sigma2)), 1e-12)
    inv = d ** (-1.0 / (M - 1.0))
    return inv / inv.sum(1, keepdims=True), d


def kfcm_J(X, V, sigma2):
    U, d = kfcm_U(X, V, sigma2)
    return float(np.sum((U**M) * d))


def kfcm_fit(X, V, sigma2, max_iter=500, tol=1e-5):
    for _ in range(max_iter):
        U, _ = kfcm_U(X, V, sigma2)
        Kw = (U**M) * np.exp(-cdist_sq_euclidean(X, V) / sigma2)
        V_new = (Kw.T @ X) / np.fmax(Kw.sum(0)[:, None], 1e-12)
        if np.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new
    return kfcm_U(X, V, sigma2)[0], V


# ------------------------------------------------------------------ một lượt chạy
def run_one(args):
    name, c, method, seed = args
    X, y = DATA[name]
    n, d = X.shape
    rng = np.random.default_rng(seed)
    model = FCM(c, m=M)
    lb, ub = X.min(0), X.max(0)
    jm = lambda V: model.objective_from_centroids(X, V)  # noqa: E731
    fe, it = 0, 0
    t0 = time.perf_counter()

    if method == "fcm":
        U, V, _ = model.fit(X, seed=rng)
    elif method == "fcm_lap":
        best = None
        for _ in range(N_RESTART):
            U_, V_, _ = model.fit(X, init_v=model.init_centroids_kmeans_plusplus(X, seed=rng), with_u=False, seed=rng)
            j = jm(V_)
            if best is None or j < best[0]:
                best = (j, U_, V_)
        _, U, V = best
    elif method == "kmeans":
        km = KMeans(c, n_init=1, random_state=int(rng.integers(1 << 31))).fit(X)
        V = km.cluster_centers_
        U = np.eye(c)[km.labels_]
    elif method in ("c1_tuan_tu", "c2_fcm_gieo", "c3_xen_ke", "c4_pso_tam"):
        P0 = init_tam(X, c, rng, model)
        if method == "c2_fcm_gieo":
            P0[0] = model.fit(X, seed=rng)[1]
        def nam_buoc_fcm(V):
            for _ in range(5):
                V = model._update_centroids(X, model._update_membership(X, V))
            return V

        hook = nam_buoc_fcm if method == "c3_xen_ke" else None
        V, _, fe, it = pso_box(P0, lb, ub, jm, rng, hook=hook)
        if method == "c4_pso_tam":
            U = model._update_membership(X, V)
        else:
            U, V, _ = model.fit(X, init_v=V, with_u=False, seed=rng)
    elif method == "c5_pso_U":
        def norm_U(U):
            U = np.fmax(U, 1e-6)
            return U / U.sum(1, keepdims=True)
        fit_U = lambda U: model.compute_objective_j(X, norm_U(U), tam_tu_U(X, norm_U(U)))  # noqa: E731
        P0 = rng.random((N_P, n, c))
        Ub, _, fe, it = pso_box(P0, np.zeros((n, c)), np.ones((n, c)), fit_U, rng)
        U = norm_U(Ub)
        V = tam_tu_U(X, U)
    elif method == "c6_pso_xb":
        fit = lambda V: xb_from_centroids(model, X, V) + phat_kich_thuoc(model._update_membership(X, V), c)  # noqa: E731
        V, _, fe, it = pso_box(init_tam(X, c, rng, model), lb, ub, fit, rng)
        U = model._update_membership(X, V)
    elif method == "c7_pso_trong_so":
        V_init = model.init_centroids_kmeans_plusplus(X, seed=rng)  # cố định trong lượt → fitness tất định theo w

        def phan_cum(w):
            s = np.sqrt(w / w.mean())
            inner = FCM(c, m=M, max_iter=200)
            U_, _, _ = inner.fit(X * s, init_v=V_init * s, with_u=False, seed=0)
            return U_, tam_tu_U(X, U_)

        def fit_w(w):
            U_, V_ = phan_cum(w)
            return xb_UV(X, U_, V_) + phat_kich_thuoc(U_, c)

        P0 = np.vstack([np.ones(d), rng.uniform(0.05, 1.0, size=(9, d))])  # 1 hạt = trọng số đều (FCM thường)
        wb, _, fe, it = pso_box(P0, np.full(d, 0.05), np.ones(d), fit_w, rng)
        U, V = phan_cum(wb)
    elif method == "c7_wfcm":
        U, V, _ = wfcm(X, c, rng)
    elif method in ("c8_pso_kfcm", "c8_kfcm"):
        sigma2 = float(d)  # z-score: tổng phương sai = d
        if method == "c8_kfcm":
            V0 = model.init_centroids_kmeans_plusplus(X, seed=rng)
        else:
            V0, _, fe, it = pso_box(init_tam(X, c, rng, model), lb, ub, lambda V: kfcm_J(X, V, sigma2), rng)
        U, V = kfcm_fit(X, V0, sigma2)
    else:
        raise ValueError(method)

    t = time.perf_counter() - t0
    return dict(dataset=name, c=c, method=method, seed=seed, t=t, fe=fe, pso_iter=it, **danh_gia(X, y, U, V, model))


if __name__ == "__main__":
    out = sys.argv[1]
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(nm, c, m, s) for nm in DATA for c in CLUSTERS for m in METHODS for s in range(n_seed)]
    print(f"Tổng {len(jobs)} lượt chạy", flush=True)
    t0 = time.perf_counter()
    with Pool(8) as pool, open(out, "w", newline="") as fh:
        writer = None
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} lượt, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"Xong sau {time.perf_counter() - t0:.0f}s → {out}")
