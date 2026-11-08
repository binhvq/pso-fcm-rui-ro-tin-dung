# -*- coding: utf-8 -*-
"""PSO làm bộ khởi tạo tâm cho FCM (thí nghiệm E5): PSO gốc và các phiên bản cải tiến.

Mỗi hạt là một bộ tâm c×D, fitness là J_m (hàm rút gọn theo tâm). Bầy khởi tạo hoàn toàn bằng c điểm dữ liệu chọn
ngẫu nhiên cho mỗi hạt (không dùng k-means++), PSO chạy ĐỦ T vòng (không dừng sớm), rồi FCM chạy tới hội tụ từ gbest.

Phiên bản:
    goc   Kennedy & Eberhart 1995: không có trọng số quán tính (ω = 1), c1 = c2 = 2, vận tốc kẹp ±vmax.
    w     Shi & Eberhart 1998: ω giảm tuyến tính 0,9 → 0,4, c1 = c2 = 2.
    s     Qian và cộng sự 2022 (Eq. 31): ω = 1,2 − 0,9·(t/T)², c1 = c2 = 2 (bài không nêu c1, c2).
    log   Qian và cộng sự 2022 (Eq. 32): ω = (ln(2,98 + t/T))^(−1,51), c1 = c2 = 2.
    qpso  Sun và cộng sự 2004, dạng của Zhang và cộng sự 2025 (Eq. 16–19): không có vận tốc, γ 1,0 → 0,1.
"""

from __future__ import annotations

from functools import partial
from time import perf_counter

import numpy as np

from fcm import FCM
from giao_dien import KetQua, tinh_chinh_fcm
from pso_co_so import omega_tuyen_tinh, pso
from thuat_toan_a import omega_LOG, omega_S
from thuat_toan_b import qpso

C_GOC = 2.0
PHIEN_BAN = ("goc", "w", "s", "log", "qpso")
OMEGA = {
    "goc": lambda t, t_max: 1.0,
    "w": omega_tuyen_tinh,
    "s": omega_S,
    "log": omega_LOG,
}


def bay_diem_ngau_nhien(X: np.ndarray, c: int, n_hat: int, rng: np.random.Generator) -> np.ndarray:
    """Bầy (n_hat, c, D): mỗi hạt là c điểm dữ liệu khác nhau chọn ngẫu nhiên."""
    return np.stack([X[rng.choice(len(X), c, replace=False)] for _ in range(n_hat)])


def pso_khoi_tao_fcm(X, c, m, rng, max_fe=None, phien_ban: str = "w", n_hat: int = 20, so_vong: int = 30) -> KetQua:
    if phien_ban not in PHIEN_BAN:
        raise ValueError(f"phiên bản PSO không hợp lệ: {phien_ban}")
    mo_hinh = FCM(c, m=m)
    t0 = perf_counter()
    bay = bay_diem_ngau_nhien(X, c, n_hat, rng)
    lb = np.broadcast_to(X.min(axis=0), bay.shape[1:])
    ub = np.broadcast_to(X.max(axis=0), bay.shape[1:])
    t1 = perf_counter()

    def fitness(V):
        return mo_hinh.objective_from_centroids(X, V)

    if phien_ban == "qpso":
        gbest, lich_su, n_fe, n_vong = qpso(bay, lb, ub, fitness, rng, max_fe, max_vong=so_vong, dung_tri_tre=False)
        j_pso = lich_su[-1]
    else:
        kq = pso(bay, lb, ub, fitness, rng, max_fe=max_fe, omega=OMEGA[phien_ban], max_vong=so_vong,
                 c1=C_GOC, c2=C_GOC, dung_tri_tre=False)
        gbest, lich_su, n_fe, n_vong, j_pso = kq.gbest, kq.lich_su, kq.n_fe, kq.n_vong, kq.gbest_f
    t2 = perf_counter()
    U, V, vong_fcm, t = tinh_chinh_fcm(X, gbest, m)
    return KetQua(U=U, V=V, lich_su=lich_su, n_fe=n_fe, n_vong_fcm=vong_fcm, t_khoi_tao=t1 - t0, t_tim_kiem=t2 - t1,
                  t_tinh_chinh=t, thong_tin=dict(phien_ban=phien_ban, n_hat=n_hat, so_vong=n_vong, j_pso=j_pso))


def fcm_diem_ngau_nhien(X, c, m, rng, max_fe=None) -> KetQua:
    """FCM từ c điểm dữ liệu ngẫu nhiên: chính là PSO khởi tạo với một hạt và không vòng lặp nào."""
    t0 = perf_counter()
    V0 = bay_diem_ngau_nhien(X, c, 1, rng)[0]
    t1 = perf_counter()
    U, V, vong_fcm, t = tinh_chinh_fcm(X, V0, m)
    return KetQua(U=U, V=V, n_vong_fcm=vong_fcm, t_khoi_tao=t1 - t0, t_tinh_chinh=t,
                  thong_tin=dict(j_pso=FCM(c, m=m).objective_from_centroids(X, V0)))


def ham_e5(phien_ban: str, n_hat: int, so_vong: int):
    return partial(pso_khoi_tao_fcm, phien_ban=phien_ban, n_hat=n_hat, so_vong=so_vong)
