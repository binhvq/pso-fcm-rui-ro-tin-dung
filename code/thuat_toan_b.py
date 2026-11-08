# -*- coding: utf-8 -*-
"""Thuật toán B: QPSO–FCM (Zhang và cộng sự 2025, PLOS ONE 20(1):e0313065).

QPSO (không có vận tốc), hạt là bộ tâm c×D, fitness J_m; sau QPSO, gbest làm tâm khởi tạo cho FCM chạy tới hội tụ
(cách "tuần tự"). Mỗi vòng t = 0..T−1:
    mbest = trung bình các pbest                                   (Eq. 16)
    φ = ω·pbest_i + (1 − ω)·gbest,  ω ~ U(0,1) từng phần tử         (Eq. 17, đã sửa lỗi in theo QPSO chuẩn)
    z = φ ± γ·|mbest − z|·ln(1/q),  q ~ U(0,1), dấu + nếu s ≥ 0,5    (Eq. 18)
    γ = 0,9·(T − t)/T + 0,1   (giảm tuyến tính 1,0 → 0,1)           (Eq. 19)
    pbest cập nhật tham lam                                        (Eq. 15)

Khác bài (ghi trong Ch.2): m = 2 (bài 1,2), bầy 20 (bài 100), T = 150 (bài 300), dừng trì trệ chung, khởi tạo bầy chung;
vị trí kẹp trong [min X, max X] vì bài không nêu biên.

Bản tiểu luận hiện tại chỉ dùng hàm qpso (với max_vong và dung_tri_tre=False) qua pso_khoi_tao.py;
qpso_fcm là phương pháp B của phiên bản trước, giữ lại để tái lập kết quả cũ.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from fcm import FCM
from giao_dien import KetQua, tinh_chinh_fcm
from pso_co_so import MAX_VONG, STALL_ITERS, STALL_TOL, khoi_tao_tam


def qpso(bay, lb, ub, fitness, rng, max_fe=None, max_vong=MAX_VONG, dung_tri_tre: bool = True):
    Z = np.clip(np.array(bay, dtype=float), lb, ub)
    n = len(Z)
    f = np.array([fitness(z) for z in Z])
    P, Pf = Z.copy(), f.copy()
    g = int(np.argmin(Pf))
    lich_su, n_fe, n_vong, tri_tre = [float(Pf[g])], n, 0, 0
    for t in range(max_vong):
        if max_fe is not None and n_fe + n > max_fe:
            break
        gamma = 0.9 * (max_vong - t) / max_vong + 0.1
        mbest = P.mean(axis=0)
        omega = rng.random(Z.shape)
        phi = omega * P + (1.0 - omega) * P[g]
        q = 1.0 - rng.random(Z.shape)  # (0, 1] để ln(1/q) hữu hạn
        dau = np.where(rng.random(Z.shape) >= 0.5, 1.0, -1.0)
        Z = np.clip(phi + dau * gamma * np.abs(mbest - Z) * np.log(1.0 / q), lb, ub)
        f = np.array([fitness(z) for z in Z])
        n_fe += n
        n_vong = t + 1
        tot_hon = f < Pf
        P[tot_hon], Pf[tot_hon] = Z[tot_hon], f[tot_hon]
        truoc = lich_su[-1]
        g = int(np.argmin(Pf))
        lich_su.append(float(Pf[g]))
        tri_tre = tri_tre + 1 if (truoc - lich_su[-1]) <= STALL_TOL * abs(truoc) else 0
        if dung_tri_tre and tri_tre >= STALL_ITERS:
            break
    return P[g].copy(), lich_su, n_fe, n_vong


def qpso_fcm(X, c, m, rng, max_fe=None) -> KetQua:
    mo_hinh = FCM(c, m=m)
    t0 = perf_counter()
    bay = khoi_tao_tam(X, c, rng)
    lb = np.broadcast_to(X.min(axis=0), bay.shape[1:])
    ub = np.broadcast_to(X.max(axis=0), bay.shape[1:])
    t1 = perf_counter()
    gbest, lich_su, n_fe, n_vong = qpso(bay, lb, ub, lambda V: mo_hinh.objective_from_centroids(X, V), rng, max_fe)
    t2 = perf_counter()
    U, V, vong_fcm, t = tinh_chinh_fcm(X, gbest, m)
    return KetQua(U=U, V=V, lich_su=lich_su, n_fe=n_fe, n_vong_fcm=vong_fcm, t_khoi_tao=t1 - t0,
                  t_tim_kiem=t2 - t1, t_tinh_chinh=t, thong_tin=dict(n_vong=n_vong))
