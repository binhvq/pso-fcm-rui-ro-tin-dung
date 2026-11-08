# -*- coding: utf-8 -*-
"""Đề xuất cải tiến: PSO–FCM có chẩn đoán (cải tiến về hiệu quả, không nhằm tìm nghiệm tốt hơn).

Ý tưởng: PSO chỉ có ích khi J_m có nhiều cực tiểu (FCM phụ thuộc khởi tạo). Vì vậy:
1. Chẩn đoán: chạy K_CHAN_DOAN lần FCM khởi tạo k-means++, lấy các giá trị J_m.
2. Nếu mọi J trùng nhau (lệch tương đối ≤ TOL_TRUNG) → J_m nhiều khả năng chỉ có 1 cực tiểu: trả nghiệm FCM tốt
   nhất, KHÔNG gọi PSO.
3. Nếu khác nhau → gieo K nghiệm FCM vào bầy PSO (phần còn lại khởi tạo như thường), PSO tối ưu J_m, dừng khi bầy trì
   trệ rồi chuyển sang FCM chạy tới hội tụ từ gbest; trả nghiệm tốt hơn giữa kết quả này và nghiệm FCM tốt nhất.

Kế thừa: FCM gieo vào bầy (Mehdizadeh và cộng sự 2008); chuyển sang FCM khi PSO trì trệ (Mane & Gaikwad 2014);
khởi tạo k-means++ (Arthur & Vassilvitskii 2007); chẩn đoán đa khởi đầu theo phát hiện P1, P16 của nhóm.
"""

from __future__ import annotations

from functools import partial
from time import perf_counter

import numpy as np

from fcm import FCM
from giao_dien import KetQua, tinh_chinh_fcm
from pso_co_so import khoi_tao_tam, pso

K_CHAN_DOAN = 5
TOL_TRUNG = 1e-4  # cùng ngưỡng gom cực tiểu với thí nghiệm chẩn đoán


def pso_fcm_chan_doan(X, c, m, rng, max_fe=None, k: int = K_CHAN_DOAN, ep_pso: bool = False) -> KetQua:
    """ep_pso=True: bỏ cổng chẩn đoán, luôn vào nhánh PSO (đo chi phí và lợi ích của nhánh PSO trên dữ liệu thật)."""
    t0 = perf_counter()
    mo_hinh = FCM(c, m=m)
    nghiem, n_vong = [], 0
    for _ in range(k):
        U, V, so_vong = mo_hinh.fit(X, init_v=mo_hinh.init_centroids_kmeans_plusplus(X, seed=rng), with_u=False)
        n_vong += so_vong
        nghiem.append((mo_hinh.objective_from_centroids(X, V), U, V))
    J = np.array([j for j, _, _ in nghiem])
    tot = int(np.argmin(J))
    nhieu_cuc_tieu = bool(J.max() - J.min() > TOL_TRUNG * abs(J.min()))
    thong_tin = dict(goi_pso=nhieu_cuc_tieu or ep_pso, J_chan_doan=J.tolist())
    if ep_pso:
        thong_tin["ep_pso"] = True
    t1 = perf_counter()
    if not thong_tin["goi_pso"]:
        return KetQua(U=nghiem[tot][1], V=nghiem[tot][2], n_vong_fcm=n_vong, t_khoi_tao=t1 - t0, thong_tin=thong_tin)

    bay = khoi_tao_tam(X, c, rng)
    bay[:k] = np.stack([V for _, _, V in nghiem])
    lb, ub = X.min(axis=0), X.max(axis=0)
    kq = pso(bay, np.broadcast_to(lb, bay.shape[1:]), np.broadcast_to(ub, bay.shape[1:]),
             lambda V: mo_hinh.objective_from_centroids(X, V), rng, max_fe=max_fe)
    t2 = perf_counter()
    U, V, so_vong, t = tinh_chinh_fcm(X, kq.gbest, m)
    if mo_hinh.objective_from_centroids(X, V) > J[tot]:
        U, V = nghiem[tot][1], nghiem[tot][2]
        thong_tin["giu_nghiem_fcm"] = True
    return KetQua(U=U, V=V, lich_su=kq.lich_su, n_fe=kq.n_fe, n_vong_fcm=n_vong + so_vong, t_khoi_tao=t1 - t0,
                  t_tim_kiem=t2 - t1, t_tinh_chinh=t, thong_tin=thong_tin)


de_xuat_ep_pso = partial(pso_fcm_chan_doan, ep_pso=True)
