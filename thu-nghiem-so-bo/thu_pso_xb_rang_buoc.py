# -*- coding: utf-8 -*-
"""Thí nghiệm thử 3: PSO tối ưu Xie–Beni có ràng buộc kích thước cụm tối thiểu.

So sánh trên bộ Việt Nam và Taiwan (m = 2):
  fcm          FCM thuần, 1 lần
  fcm_lap_J    FCM chạy lại R lần (khởi tạo k-means++), chọn nghiệm J nhỏ nhất
  fcm_lap_XB   FCM chạy lại R lần, chọn nghiệm XB nhỏ nhất (đối chứng: chỉ cần chạy lại + chọn theo XB?)
  pso_xb       PSO tối ưu XB, không ràng buộc
  pso_xb_rb5   PSO tối ưu XB + phạt khi cụm mờ nhỏ nhất < 5% dữ liệu
  pso_xb_rb10  như trên, ngưỡng 10%

Chạy: python3 thu_pso_xb_rang_buoc.py <file CSV> [số seed]
"""

from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

import thu_pso_sau as nen
from thu_pso_sau import DATA, FCM, danh_gia, pso, xb_from_centroids

M = nen.M
CLUSTERS = (2, 3, 4, 5)
N_RESTART = 20
METHODS = ("fcm", "fcm_lap_J", "fcm_lap_XB", "pso_xb", "pso_xb_rb5", "pso_xb_rb10")
NGUONG = {"pso_xb_rb5": 0.05, "pso_xb_rb10": 0.10}


def xb_rang_buoc(model, X, V, nguong):
    """XB + phạt tuyến tính khi cụm cứng nhỏ nhất (tỉ lệ điểm có độ thuộc cao nhất vào cụm) dưới ngưỡng.

    Không dùng kích thước cụm mờ Σ_i u_ik / n: với m = 2 độ thuộc trải đều nên kích thước mờ
    luôn lớn dù cụm cứng gần rỗng, phạt không bao giờ kích hoạt (đã thử).
    """
    U = model._update_membership(X, V)
    nho_nhat = np.bincount(U.argmax(1), minlength=V.shape[0]).min() / len(X)
    return xb_from_centroids(model, X, V) + max(0.0, nguong - nho_nhat) / nguong


def fcm_lap_lai(X, c, rng, chon_theo):
    model = FCM(c, m=M)
    best = None
    for _ in range(N_RESTART):
        V0 = model.init_centroids_kmeans_plusplus(X, seed=rng)
        U, V, _ = model.fit(X, init_v=V0, with_u=False, seed=rng)
        score = model.objective_from_centroids(X, V) if chon_theo == "J" else xb_from_centroids(model, X, V)
        if best is None or score < best[0]:
            best = (score, U, V)
    return best[1], best[2], model


def run_one(args):
    name, c, method, seed = args
    X, y = DATA[name]
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    fe, it_pso = 0, 0
    if method == "fcm":
        model = FCM(c, m=M)
        U, V, _ = model.fit(X, seed=rng)
    elif method in ("fcm_lap_J", "fcm_lap_XB"):
        U, V, model = fcm_lap_lai(X, c, rng, "J" if method == "fcm_lap_J" else "XB")
    else:
        model = FCM(c, m=M)
        if method == "pso_xb":
            fitness = lambda V: xb_from_centroids(model, X, V)  # noqa: E731
        else:
            fitness = lambda V: xb_rang_buoc(model, X, V, NGUONG[method])  # noqa: E731
        V, fe, it_pso = pso(X, c, rng, model, fitness)
        U = model._update_membership(X, V)
    t = time.perf_counter() - t0
    return dict(dataset=name, c=c, method=method, seed=seed, t=t, fe=fe, pso_iter=it_pso, **danh_gia(X, y, U, V, model))


if __name__ == "__main__":
    out = sys.argv[1]
    n_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    jobs = [(n, c, m, s) for n in DATA for c in CLUSTERS for m in METHODS for s in range(n_seed)]
    print(f"Tổng {len(jobs)} lượt chạy", flush=True)
    t0 = time.perf_counter()
    with Pool(8) as pool, open(out, "w", newline="") as fh:
        writer = None
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=2), 1):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} lượt, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"Xong sau {time.perf_counter() - t0:.0f}s → {out}")
