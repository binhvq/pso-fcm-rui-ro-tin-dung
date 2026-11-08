# -*- coding: utf-8 -*-
"""Thí nghiệm thử: khởi tạo quần thể hạt có cải tiến có làm PSO–FCM nhanh hơn?

Giả thiết: khởi tạo vị trí hạt (bộ tâm cụm ban đầu) bằng tent map + k-means++ +
học đối lập giúp PSO–FCM hai pha hội tụ sau ít lần tính hàm mục tiêu hơn và tổng
thời gian chạy ngắn hơn khởi tạo ngẫu nhiên, ở cùng chất lượng nghiệm (J_m).

Chạy: python3 thu_gia_thiet_khoi_tao.py <thư mục code> <thư mục data/raw> <file CSV kết quả> [số seed]
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

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from fcm import FCM  # noqa: E402
import chi_so_danh_gia as cs  # noqa: E402

RAW = sys.argv[2]
OUT = sys.argv[3]
N_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20

M = 2.0
N_PARTICLES = 20
MAX_ITER = 150
STALL_TOL = 1e-5
STALL_ITERS = 10
W_START, W_END = 0.9, 0.4
C1 = C2 = 1.49445
CLUSTERS = (2, 3, 4)
METHODS = ("fcm_thuan", "ngau_nhien_deu", "diem_du_lieu", "tent", "kmeanspp", "doi_lap", "de_xuat")


# ---------------------------------------------------------------- dữ liệu
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
    for col in ("X6", "X7", "X12"):  # số 0 là missing giả
        feats[col] = feats[col].replace(0, np.nan)
        feats[col] = feats[col].fillna(feats[col].median())
    feats = feats.drop(columns=["X15", "X18", "X21", "X3"])  # gần trùng (|r| > 0,94)
    out["viet_nam"] = (zscore(signed_log(feats.values.astype(float))), v["Financial_Distress"].values)

    t = pd.read_excel(glob.glob(RAW + "/taiwan-default/*.xls")[0], header=1)
    y = t.iloc[:, -1].values
    X = t.drop(columns=[t.columns[0], t.columns[-1]]).values.astype(float)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
    out["taiwan_5000"] = (zscore(signed_log(X[idx])), y[idx])

    s = pd.read_csv(RAW + "/south-german-credit/SouthGermanCredit.asc", sep=r"\s+")
    out["south_german"] = (zscore(s.drop(columns="kredit").values.astype(float)), s["kredit"].values)

    g = np.loadtxt(RAW + "/german-credit/german.data-numeric")
    out["german_so"] = (zscore(g[:, :-1]), g[:, -1])

    a = np.loadtxt(RAW + "/australian-credit/australian.dat")
    out["australian"] = (zscore(signed_log(a[:, :-1])), a[:, -1])
    return out


# ---------------------------------------------------------------- khởi tạo
def tent_sequence(n, rng):
    """Dãy hỗn loạn tent map trong (0, 1); gieo lại khi rơi vào điểm cố định/chu kỳ ngắn."""
    z = np.empty(n)
    x = rng.uniform(0.01, 0.99)
    for i in range(n):
        x = 2 * x if x < 0.5 else 2 * (1 - x)
        if x < 1e-6 or abs(x - 0.5) < 1e-6 or abs(x - 1) < 1e-6:
            x = rng.uniform(0.01, 0.99)
        z[i] = x
    return z


def init_swarm(method, X, c, rng, model, fitness):
    """Trả về (vị trí (N, c, D), fitness (N,), số FE đã dùng)."""
    n, d = X.shape
    lb, ub = X.min(0), X.max(0)
    N = N_PARTICLES

    def uniform():
        return rng.uniform(lb, ub, size=(N, c, d))

    def data_points():
        return np.stack([X[rng.choice(n, c, replace=False)] for _ in range(N)])

    def tent():
        return lb + tent_sequence(N * c * d, rng).reshape(N, c, d) * (ub - lb)

    def kpp(k):
        return np.stack([model.init_centroids_kmeans_plusplus(X, seed=rng) for _ in range(k)])

    def opposite_select(P):
        Q = np.clip(lb + ub - P, lb, ub)
        both = np.concatenate([P, Q])
        f = np.array([fitness(p) for p in both])
        best = np.argsort(f)[:N]
        return both[best], f[best], 2 * N

    if method == "ngau_nhien_deu":
        P = uniform()
    elif method == "diem_du_lieu":
        P = data_points()
    elif method == "tent":
        P = tent()
    elif method == "kmeanspp":
        P = kpp(N)
    elif method == "doi_lap":
        return opposite_select(uniform())
    elif method == "de_xuat":
        P = np.concatenate([kpp(N // 2), tent()[: N - N // 2]])
        return opposite_select(P)
    else:
        raise ValueError(method)
    return P, np.array([fitness(p) for p in P]), N


# ---------------------------------------------------------------- PSO–FCM hai pha
def pso_fcm(X, c, method, seed):
    rng = np.random.default_rng(seed)
    model = FCM(c, m=M)
    fitness = lambda V: model.objective_from_centroids(X, V)  # noqa: E731
    lb, ub = X.min(0), X.max(0)
    vmax = 0.2 * (ub - lb)

    t0 = time.perf_counter()
    P, f, fe = init_swarm(method, X, c, rng, model, fitness)
    t_init = time.perf_counter() - t0

    t1 = time.perf_counter()
    Vel = np.zeros_like(P)
    pbest, pbest_f = P.copy(), f.copy()
    g = int(np.argmin(f))
    gbest, gbest_f = P[g].copy(), f[g]
    stall, it = 0, 0
    trace = [(fe, gbest_f)]  # (FE tích luỹ, gbest) sau khởi tạo và sau mỗi vòng
    for it in range(1, MAX_ITER + 1):
        w = W_START - (W_START - W_END) * (it - 1) / (MAX_ITER - 1)
        r1, r2 = rng.random(P.shape), rng.random(P.shape)
        Vel = np.clip(w * Vel + C1 * r1 * (pbest - P) + C2 * r2 * (gbest - P), -vmax, vmax)
        P = np.clip(P + Vel, lb, ub)
        f = np.array([fitness(p) for p in P])
        fe += len(P)
        better = f < pbest_f
        pbest[better], pbest_f[better] = P[better], f[better]
        g = int(np.argmin(pbest_f))
        prev = gbest_f
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        trace.append((fe, gbest_f))
        stall = stall + 1 if (prev - gbest_f) <= STALL_TOL * abs(prev) else 0
        if stall >= STALL_ITERS:
            break
    t_pso = time.perf_counter() - t1

    U, V, fcm_iter = model.fit(X, init_v=gbest, with_u=False, seed=rng)
    return dict(fe=fe, pso_iter=it, fcm_iter=fcm_iter, t_init=t_init, t_pso=t_pso, t_fcm=model.process_time,
                J_gbest=gbest_f, U=U, V=V, model=model,
                trace=";".join(f"{a}:{b:.6f}" for a, b in trace))


def fcm_thuan(X, c, seed):
    model = FCM(c, m=M)
    U, V, it = model.fit(X, seed=seed)
    return dict(fe=0, pso_iter=0, fcm_iter=it, t_init=0.0, t_pso=0.0, t_fcm=model.process_time,
                J_gbest=np.nan, U=U, V=V, model=model, trace="")


def run_one(args):
    name, c, method, seed = args
    X, y = DATA[name]
    r = fcm_thuan(X, c, seed) if method == "fcm_thuan" else pso_fcm(X, c, method, seed)
    U, V, model = r.pop("U"), r.pop("V"), r.pop("model")
    labels = U.argmax(1)
    rates = [float(np.mean(y[labels == k] == y.max())) if np.any(labels == k) else np.nan for k in range(c)]
    return dict(dataset=name, c=c, method=method, seed=seed, **r,
                t_total=r["t_init"] + r["t_pso"] + r["t_fcm"],
                J=model.objective_from_centroids(X, V), PC=cs.pc(U), XB=cs.xb(X, V, U, M),
                sup=bool(np.abs(V).max() < 1e-3), ty_le_nhan_max=max(rates), ty_le_nhan_min=min(rates))


DATA = load_datasets()

if __name__ == "__main__":
    for k, (X, y) in DATA.items():
        print(f"{k}: {X.shape}, tỉ lệ nhãn dương {np.mean(y == y.max()):.3f}", flush=True)
    jobs = [(name, c, m, s) for name in DATA for c in CLUSTERS for m in METHODS for s in range(N_SEED)]
    print(f"Tổng {len(jobs)} lượt chạy", flush=True)
    t0 = time.perf_counter()
    with Pool(8) as pool, open(OUT, "w", newline="") as fh:
        writer = None
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} lượt, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"Xong sau {time.perf_counter() - t0:.0f}s → {OUT}")
