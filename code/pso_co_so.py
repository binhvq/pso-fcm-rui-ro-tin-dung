# -*- coding: utf-8 -*-
"""PSO hộp giới hạn dùng chung cho mọi cách kết hợp PSO–FCM.

Mỗi hạt là một mảng bất kỳ (bộ tâm c×D, vector trọng số D, ma trận độ thuộc n×c). Cập nhật chuẩn
(Kennedy & Eberhart 1995; Shi & Eberhart 1998):

    v ← ω(t)·v + c1·r1·(pbest − x) + c2·r2·(gbest − x),   v kẹp trong ±vmax,   x ← kẹp(x + v, lb, ub)

ω(t) là hàm truyền vào (mặc định giảm tuyến tính 0,9 → 0,4). Dừng khi đạt số vòng tối đa, khi gbest không cải thiện
quá STALL_TOL (tương đối) trong STALL_ITERS vòng liên tiếp, hoặc khi vòng kế tiếp sẽ vượt ngân sách max_fe.

Các giá trị mặc định dưới đây (c1 = c2 = 1,49445, 150 vòng, dừng trì trệ) thuộc phiên bản trước; pso_khoi_tao.py
truyền tường minh c1 = c2 = 2, lịch ω, số vòng và dung_tri_tre=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fcm import FCM

N_HAT = 20
MAX_VONG = 150
C1 = C2 = 1.49445
W_DAU, W_CUOI = 0.9, 0.4
VMAX_TI_LE = 0.2
STALL_TOL, STALL_ITERS = 1e-5, 30  # cửa sổ 10 vòng dừng sớm ở 22/40 lượt trên hàm cầu 4 chiều; 30 vòng: 0/40


def omega_tuyen_tinh(t: int, t_max: int, w_dau: float = W_DAU, w_cuoi: float = W_CUOI) -> float:
    """ω giảm tuyến tính từ w_dau (t = 1) tới w_cuoi (t = t_max)."""
    return w_dau - (w_dau - w_cuoi) * (t - 1) / max(t_max - 1, 1)


@dataclass
class KetQuaPSO:
    gbest: np.ndarray
    gbest_f: float
    lich_su: list[float] = field(default_factory=list)  # gbest_f sau khởi tạo và sau mỗi vòng
    n_fe: int = 0
    n_vong: int = 0


def pso(khoi_tao: np.ndarray, lb, ub, fitness: Callable[[np.ndarray], float], rng: np.random.Generator,
        max_fe: int | None = None, omega: Callable[[int, int], float] = omega_tuyen_tinh, max_vong: int = MAX_VONG,
        c1: float = C1, c2: float = C2, sau_cap_nhat: Callable[[np.ndarray, int], np.ndarray] | None = None,
        dung_khi: Callable[[KetQuaPSO], bool] | None = None, dung_tri_tre: bool = True) -> KetQuaPSO:
    """Chạy PSO tối thiểu hoá `fitness` từ bầy `khoi_tao` (N, ...).

    sau_cap_nhat(hat, vong): biến đổi vị trí mỗi hạt sau bước di chuyển (ví dụ vài bước FCM ở cách xen kẽ).
    dung_khi(trang_thai): điều kiện dừng sớm thêm (ví dụ chuyển sang FCM khi bầy trì trệ).
    dung_tri_tre=False: chạy đủ max_vong (cho thuật toán mà bài gốc không có dừng sớm).
    """
    P = np.array(khoi_tao, dtype=float, copy=True)
    n = len(P)
    if max_fe is not None and max_fe < n:
        raise ValueError(f"max_fe = {max_fe} nhỏ hơn số hạt {n}")
    lb = np.broadcast_to(np.asarray(lb, dtype=float), P.shape[1:])
    ub = np.broadcast_to(np.asarray(ub, dtype=float), P.shape[1:])
    vmax = VMAX_TI_LE * (ub - lb)
    P = np.clip(P, lb, ub)
    f = np.array([fitness(p) for p in P])
    pb, pbf = P.copy(), f.copy()
    g = int(np.argmin(f))
    kq = KetQuaPSO(gbest=P[g].copy(), gbest_f=float(f[g]), lich_su=[float(f[g])], n_fe=n)
    V = np.zeros_like(P)
    tri_tre = 0
    for t in range(1, max_vong + 1):
        if max_fe is not None and kq.n_fe + n > max_fe:
            break
        w = omega(t, max_vong)
        V = w * V + c1 * rng.random(P.shape) * (pb - P) + c2 * rng.random(P.shape) * (kq.gbest - P)
        V = np.clip(V, -vmax, vmax)
        P = np.clip(P + V, lb, ub)
        if sau_cap_nhat is not None:
            P = np.stack([sau_cap_nhat(p, t) for p in P])
        f = np.array([fitness(p) for p in P])
        kq.n_fe += n
        kq.n_vong = t
        tot_hon = f < pbf
        pb[tot_hon], pbf[tot_hon] = P[tot_hon], f[tot_hon]
        truoc = kq.gbest_f
        g = int(np.argmin(pbf))
        if pbf[g] < kq.gbest_f:
            kq.gbest, kq.gbest_f = pb[g].copy(), float(pbf[g])
        kq.lich_su.append(kq.gbest_f)
        tri_tre = tri_tre + 1 if (truoc - kq.gbest_f) <= STALL_TOL * abs(truoc) else 0
        if (dung_tri_tre and tri_tre >= STALL_ITERS) or (dung_khi is not None and dung_khi(kq)):
            break
    return kq


def khoi_tao_tam(X: np.ndarray, c: int, rng: np.random.Generator, n_hat: int = N_HAT) -> np.ndarray:
    """Bầy hạt là bộ tâm (n_hat, c, D): nửa khởi tạo k-means++, nửa chọn c điểm dữ liệu ngẫu nhiên."""
    mo_hinh = FCM(c)
    nua = n_hat // 2
    kpp = [mo_hinh.init_centroids_kmeans_plusplus(X, seed=rng) for _ in range(nua)]
    diem = [X[rng.choice(len(X), c, replace=False)] for _ in range(n_hat - nua)]
    return np.stack(kpp + diem)
