# -*- coding: utf-8 -*-
"""Thí nghiệm thử: tích hợp PSO sâu hơn khởi tạo tâm cụm.

So sánh trên bộ Việt Nam và Taiwan (m = 2):
  fcm          FCM thuần (đối chứng)
  pso_jm       PSO tối ưu J_m rồi FCM tinh chỉnh từ gbest (thuật toán A, đối chứng)
  memetic      PSO–FCM lai: mỗi vòng, mỗi hạt chạy thêm 1 bước FCM (Lamarck)
  pso_xb       PSO tối ưu Xie–Beni trên bộ tâm (không tinh chỉnh FCM)
  pso_xb_fcm   PSO tối ưu Xie–Beni rồi FCM tinh chỉnh (kiểm tra FCM có kéo về nghiệm J_m)

Nhãn vỡ nợ chỉ dùng để đánh giá khả năng tách rủi ro, không đưa vào thuật toán.

Chạy: python3 thu_pso_sau.py <file CSV> [số seed]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import glob
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, silhouette_score

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "code"))
from fcm import FCM, cdist_sq_euclidean  # noqa: E402
import chi_so_danh_gia as cs  # noqa: E402

RAW = str(GOC / "data" / "raw")

M = 2.0
N_PARTICLES = 20
MAX_ITER = 150
STALL_TOL, STALL_ITERS = 1e-5, 10
W_START, W_END = 0.9, 0.4
C1 = C2 = 1.49445
CLUSTERS = (2, 3, 4, 5)
METHODS = ("fcm", "pso_jm", "memetic", "pso_xb", "pso_xb_fcm")


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def zscore(X):
    sd = X.std(0)
    keep = sd > 0
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]


def load_datasets():
    out = {}
    v = pd.read_csv(glob.glob(RAW + "/vn-financial-distress-listed-firms/*.csv")[0], encoding="latin-1")
    feats = v[[f"X{i}" for i in range(1, 26)]].copy()
    for col in ("X6", "X7", "X12"):
        feats[col] = feats[col].replace(0, np.nan)
        feats[col] = feats[col].fillna(feats[col].median())
    feats = feats.drop(columns=["X15", "X18", "X21", "X3"])
    out["viet_nam"] = (zscore(signed_log(feats.values.astype(float))), v["Financial_Distress"].values)

    t = pd.read_excel(glob.glob(RAW + "/taiwan-default/*.xls")[0], header=1)
    y = t.iloc[:, -1].values
    X = t.drop(columns=[t.columns[0], t.columns[-1]]).values.astype(float)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
    out["taiwan_5000"] = (zscore(signed_log(X[idx])), y[idx])
    return out


def xb_from_centroids(model, X, V):
    """Xie–Beni của bộ tâm V với U tối ưu theo V (m = 2)."""
    costs = model._compute_cost_matrix(X, V)
    U = model._cost_to_membership(costs)
    sep = cdist_sq_euclidean(V, V)
    np.fill_diagonal(sep, np.inf)
    return float(np.sum((U**M) * costs) / (len(X) * max(sep.min(), 1e-12)))


def init_swarm(X, c, rng, model):
    """Nửa hạt k-means++, nửa hạt điểm dữ liệu ngẫu nhiên."""
    half = N_PARTICLES // 2
    kpp = [model.init_centroids_kmeans_plusplus(X, seed=rng) for _ in range(half)]
    pts = [X[rng.choice(len(X), c, replace=False)] for _ in range(N_PARTICLES - half)]
    return np.stack(kpp + pts)


def pso(X, c, rng, model, fitness, memetic=False):
    lb, ub = X.min(0), X.max(0)
    vmax = 0.2 * (ub - lb)
    P = init_swarm(X, c, rng, model)
    if memetic:
        P = np.stack([model._update_centroids(X, model._update_membership(X, p)) for p in P])
    f = np.array([fitness(p) for p in P])
    fe = len(P)
    Vel = np.zeros_like(P)
    pbest, pbest_f = P.copy(), f.copy()
    g = int(np.argmin(f))
    gbest, gbest_f = P[g].copy(), f[g]
    stall, it = 0, 0
    for it in range(1, MAX_ITER + 1):
        w = W_START - (W_START - W_END) * (it - 1) / (MAX_ITER - 1)
        r1, r2 = rng.random(P.shape), rng.random(P.shape)
        Vel = np.clip(w * Vel + C1 * r1 * (pbest - P) + C2 * r2 * (gbest - P), -vmax, vmax)
        P = np.clip(P + Vel, lb, ub)
        if memetic:  # một bước FCM cho mỗi hạt, ghi đè vị trí (học kiểu Lamarck)
            P = np.stack([model._update_centroids(X, model._update_membership(X, p)) for p in P])
        f = np.array([fitness(p) for p in P])
        fe += len(P)
        better = f < pbest_f
        pbest[better], pbest_f[better] = P[better], f[better]
        g = int(np.argmin(pbest_f))
        prev = gbest_f
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        stall = stall + 1 if (prev - gbest_f) <= STALL_TOL * abs(prev) else 0
        if stall >= STALL_ITERS:
            break
    return gbest, fe, it


def danh_gia(X, y, U, V, model):
    """Chỉ số chất lượng cụm và khả năng tách rủi ro (nhãn chỉ dùng để đánh giá)."""
    c = V.shape[0]
    labels = U.argmax(1)
    pos = y == y.max()
    rates = np.array([pos[labels == k].mean() if np.any(labels == k) else np.nan for k in range(c)])
    risky = int(np.nanargmax(rates))
    sample = np.random.default_rng(0).choice(len(X), min(len(X), 3000), replace=False)
    sil = silhouette_score(X[sample], labels[sample]) if len(np.unique(labels[sample])) > 1 else np.nan
    return dict(J=model.objective_from_centroids(X, V), XB=xb_from_centroids(model, X, V), PC=cs.pc(U), PE=cs.pe(U),
                silhouette=sil, so_cum_rong=int(np.sum(~np.isnan(rates))),
                cum_nho_nhat=float(np.bincount(labels, minlength=c).min() / len(X)),
                chenh_ty_le=float(np.nanmax(rates) - np.nanmin(rates)),
                auc_cum_rui_ro=float(roc_auc_score(pos, U[:, risky])))


def run_one(args):
    name, c, method, seed = args
    X, y = DATA[name]
    rng = np.random.default_rng(seed)
    model = FCM(c, m=M)
    jm = lambda V: model.objective_from_centroids(X, V)  # noqa: E731
    xb = lambda V: xb_from_centroids(model, X, V)  # noqa: E731

    t0 = time.perf_counter()
    fe, it_pso = 0, 0
    if method == "fcm":
        U, V, _ = model.fit(X, seed=rng)
    else:
        if method == "pso_jm":
            V, fe, it_pso = pso(X, c, rng, model, jm)
        elif method == "memetic":
            V, fe, it_pso = pso(X, c, rng, model, jm, memetic=True)
        else:
            V, fe, it_pso = pso(X, c, rng, model, xb)
        if method in ("pso_jm", "memetic", "pso_xb_fcm"):
            U, V, _ = model.fit(X, init_v=V, with_u=False, seed=rng)
        else:
            U = model._update_membership(X, V)
    t = time.perf_counter() - t0
    return dict(dataset=name, c=c, method=method, seed=seed, t=t, fe=fe, pso_iter=it_pso, **danh_gia(X, y, U, V, model))


DATA = load_datasets()

if __name__ == "__main__":
    OUT = sys.argv[1]
    N_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(n, c, m, s) for n in DATA for c in CLUSTERS for m in METHODS for s in range(N_SEED)]
    print(f"Tổng {len(jobs)} lượt chạy", flush=True)
    t0 = time.perf_counter()
    with Pool(8) as pool, open(OUT, "w", newline="") as fh:
        writer = None
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=2), 1):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} lượt, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"Xong sau {time.perf_counter() - t0:.0f}s → {OUT}")
