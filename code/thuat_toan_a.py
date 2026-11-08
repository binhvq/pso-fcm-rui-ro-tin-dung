# -*- coding: utf-8 -*-
"""Thuật toán A: PSO–FCM với trọng số quán tính biến thiên (Qian và cộng sự 2022, Remote Sensing 14:3713).

Hạt là bộ tâm c×D; fitness J_m (Eq. 35); U suy từ tâm theo công thức độ thuộc FCM; không có bước FCM chạy sau PSO
(bài không nêu) — tức thuộc cách "PSO thay FCM, hạt là tâm". Ba kiểu trọng số quán tính, i/maxgen với i = 1..maxgen:
    L   (Eq. 30)  ω = ws − (ws − we)·(i/maxgen)
    S   (Eq. 31)  ω = ws − (ws − we)·(i/maxgen)²
    LOG (Eq. 32)  ω = (ln(2,98 + i/maxgen))^(−z), z = 1,51
với ws = 1,2, we = 0,3. Bài kết luận S tốt nhất (tr. 21), nhưng Bảng 5 cho thấy điều đó chỉ đúng ở 1000 vòng.

Chạy đủ số vòng như bài (không dừng sớm): với ws = 1,2 > 1 bầy bung ra ở giai đoạn đầu nên gbest có thể đứng yên
hàng chục vòng; quy tắc dừng trì trệ chung sẽ cắt thuật toán trước khi ω giảm.

Khác bài (để so sánh công bằng, ghi trong Ch.2): m = 2, bầy 20 hạt, tối đa 150 vòng, khởi tạo bầy
chung thay cho khởi tạo bằng đồ thị (bước đó tự chọn số cụm, còn tiểu luận cố định c); c1 = c2 = 1,49445 và vmax = 20%
miền vì bài không nêu.

Bản tiểu luận hiện tại chỉ dùng omega_S và omega_LOG (lịch quán tính PSO-S, PSO-LOG) qua pso_khoi_tao.py;
thuật toán A đầy đủ là phương pháp của phiên bản trước, giữ lại để tái lập kết quả cũ.
"""

from __future__ import annotations

from functools import partial
from time import perf_counter

import numpy as np

from fcm import FCM
from giao_dien import KetQua
from pso_co_so import khoi_tao_tam, pso

WS, WE, Z_LOG = 1.2, 0.3, 1.51


def omega_L(t, t_max):
    return WS - (WS - WE) * (t / t_max)


def omega_S(t, t_max):
    return WS - (WS - WE) * (t / t_max) ** 2


def omega_LOG(t, t_max):
    return np.log(2.98 + t / t_max) ** (-Z_LOG)


OMEGA = {"L": omega_L, "S": omega_S, "LOG": omega_LOG}


def pso_fcm_qian(X, c, m, rng, max_fe=None, kieu: str = "S") -> KetQua:
    mo_hinh = FCM(c, m=m)
    t0 = perf_counter()
    bay = khoi_tao_tam(X, c, rng)
    lb = np.broadcast_to(X.min(axis=0), bay.shape[1:])
    ub = np.broadcast_to(X.max(axis=0), bay.shape[1:])
    t1 = perf_counter()
    kq = pso(bay, lb, ub, lambda V: mo_hinh.objective_from_centroids(X, V), rng, max_fe=max_fe, omega=OMEGA[kieu],
             dung_tri_tre=False)
    U = mo_hinh._update_membership(X, kq.gbest)
    return KetQua(U=U, V=kq.gbest, lich_su=kq.lich_su, n_fe=kq.n_fe, t_khoi_tao=t1 - t0,
                  t_tim_kiem=perf_counter() - t1, thong_tin=dict(kieu_omega=kieu, n_vong=kq.n_vong))


a_L = partial(pso_fcm_qian, kieu="L")
a_S = partial(pso_fcm_qian, kieu="S")
a_LOG = partial(pso_fcm_qian, kieu="LOG")
