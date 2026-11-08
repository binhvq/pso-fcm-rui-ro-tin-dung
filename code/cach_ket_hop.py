# -*- coding: utf-8 -*-
"""Các cách kết hợp PSO với FCM trong tài liệu và các đối chứng, cùng giao diện fit(X, c, m, rng, max_fe) → KetQua.

Loại I — PSO tối ưu cùng hàm J_m với FCM:
    cach2_fcm_gieo   FCM chạy trước, nghiệm FCM là một hạt trong bầy (Mehdizadeh và cộng sự 2008)
    cach3_xen_ke     cứ 10 vòng PSO, mỗi hạt chạy 5 bước FCM (memetic; Izakian & Abraham 2011)
    (cách 4 "PSO thay FCM, hạt là tâm" chính là thuật toán A, bài 02 — module thuat_toan_a)
    cach5_pso_U      PSO thay FCM, hạt là ma trận độ thuộc U (Runkler & Katz 2006; bài 05)
    (cách 1 "tuần tự" là thuật toán B — module thuat_toan_b; C ở module thuat_toan_c)
Loại II — PSO tối ưu thứ FCM không tự tối ưu được:
    cach7_pso_trong_so   PSO tìm trọng số thuộc tính, FCM phân cụm trên dữ liệu đã nhân trọng số; fitness Silhouette
                         trên không gian gốc (O'Neill và cộng sự 2018; Lu và cộng sự 2011)
    cach8_pso_kfcm       PSO tìm tâm cho kernel FCM (nhân Gauss), KFCM tinh chỉnh (bài 03)
Đối chứng: fcm_lap_lai (FCM chạy lại 20 lần, chọn J nhỏ nhất), wfcm_giai_tich (trọng số toàn cục tính bằng công thức, đối chứng cho cách 7),
kfcm (kernel FCM không PSO).

Mọi cách kết thúc bằng FCM chạy tới hội tụ (tinh_chinh_fcm) trừ cách 5 (theo đúng định nghĩa "PSO thay FCM").
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import silhouette_score

from fcm import FCM, cdist_euclidean, cdist_sq_euclidean
from giao_dien import KetQua, tinh_chinh_fcm
from pso_co_so import N_HAT, khoi_tao_tam, pso

SO_LAN_LAP = 20
NGUONG_CUM = 0.05  # cụm cứng nhỏ nhất ≥ 5% dữ liệu (phân khúc nhỏ hơn không đủ để có chính sách riêng)
N_HAT_TRONG_SO = 10
VONG_FCM_TRONG = 200  # trần vòng FCM khi chấm một hạt trọng số; phân hoạch cuối chạy tới hội tụ
N_MAU_SILHOUETTE = 1000


def tam_tu_U(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    W = U**m
    return (W.T @ X) / np.fmax(W.sum(axis=0)[:, None], 1e-12)


def phat_kich_thuoc(U: np.ndarray) -> float:
    """0 nếu cụm cứng nhỏ nhất ≥ NGUONG_CUM; tăng tuyến tính tới 1 khi cụm nhỏ nhất về 0."""
    c = U.shape[1]
    nho = np.bincount(U.argmax(axis=1), minlength=c).min() / len(U)
    return max(0.0, NGUONG_CUM - nho) / NGUONG_CUM


def _gioi_han(X):
    return X.min(axis=0), X.max(axis=0)


# ------------------------------------------------------------------ đối chứng
def fcm_lap_lai(X, c, m, rng, max_fe=None) -> KetQua:
    """FCM chạy lại SO_LAN_LAP lần (khởi tạo k-means++), chọn nghiệm có J_m nhỏ nhất."""
    t0 = perf_counter()
    mo_hinh, tot, n_vong = FCM(c, m=m), None, 0
    for _ in range(SO_LAN_LAP):
        U, V, k = mo_hinh.fit(X, init_v=mo_hinh.init_centroids_kmeans_plusplus(X, seed=rng), with_u=False)
        n_vong += k
        j = mo_hinh.objective_from_centroids(X, V)
        if tot is None or j < tot[0]:
            tot = (j, U, V)
    return KetQua(U=tot[1], V=tot[2], n_vong_fcm=n_vong, t_tinh_chinh=perf_counter() - t0)


# ------------------------------------------------------------------ loại I
def _pso_tam_theo_jm(X, c, m, rng, max_fe, bay, **kw):
    mo_hinh = FCM(c, m=m)
    lb, ub = _gioi_han(X)
    lb, ub = np.broadcast_to(lb, (c, X.shape[1])), np.broadcast_to(ub, (c, X.shape[1]))
    return pso(bay, lb, ub, lambda V: mo_hinh.objective_from_centroids(X, V), rng, max_fe=max_fe, **kw)


def _tinh_chinh(X, m, V0, kq_pso, t_khoi_tao, t_tim_kiem) -> KetQua:
    U, V, n_vong, t = tinh_chinh_fcm(X, V0, m)
    return KetQua(U=U, V=V, lich_su=kq_pso.lich_su, n_fe=kq_pso.n_fe, n_vong_fcm=n_vong,
                  t_khoi_tao=t_khoi_tao, t_tim_kiem=t_tim_kiem, t_tinh_chinh=t)


def cach2_fcm_gieo(X, c, m, rng, max_fe=None) -> KetQua:
    t0 = perf_counter()
    bay = khoi_tao_tam(X, c, rng)
    bay[0] = FCM(c, m=m).fit(X, seed=rng)[1]
    t1 = perf_counter()
    kq = _pso_tam_theo_jm(X, c, m, rng, max_fe, bay)
    return _tinh_chinh(X, m, kq.gbest, kq, t1 - t0, perf_counter() - t1)


def cach3_xen_ke(X, c, m, rng, max_fe=None, chu_ky: int = 10, so_buoc_fcm: int = 5) -> KetQua:
    mo_hinh = FCM(c, m=m)

    def vai_buoc_fcm(V, vong):
        if vong % chu_ky:
            return V
        for _ in range(so_buoc_fcm):
            V = mo_hinh._update_centroids(X, mo_hinh._update_membership(X, V))
        return V

    t0 = perf_counter()
    bay = khoi_tao_tam(X, c, rng)
    t1 = perf_counter()
    kq = _pso_tam_theo_jm(X, c, m, rng, max_fe, bay, sau_cap_nhat=vai_buoc_fcm)
    return _tinh_chinh(X, m, kq.gbest, kq, t1 - t0, perf_counter() - t1)


def cach5_pso_U(X, c, m, rng, max_fe=None) -> KetQua:
    mo_hinh = FCM(c, m=m)

    def chuan_hoa(U):
        U = np.fmax(U, 1e-6)
        return U / U.sum(axis=1, keepdims=True)

    def j_theo_U(U):
        U = chuan_hoa(U)
        return mo_hinh.compute_objective_j(X, U, tam_tu_U(X, U, m))

    t0 = perf_counter()
    bay = rng.random((N_HAT, len(X), c))
    t1 = perf_counter()
    kq = pso(bay, 0.0, 1.0, j_theo_U, rng, max_fe=max_fe)
    U = chuan_hoa(kq.gbest)
    return KetQua(U=U, V=tam_tu_U(X, U, m), lich_su=kq.lich_su, n_fe=kq.n_fe, t_khoi_tao=t1 - t0,
                  t_tim_kiem=perf_counter() - t1)


# ------------------------------------------------------------------ loại II: trọng số thuộc tính
def wfcm_giai_tich(X, c, m, rng, max_fe=None, beta: float = 2.0, max_vong: int = 300, tol: float = 1e-5) -> KetQua:
    """FCM có trọng số thuộc tính toàn cục D chiều tính bằng công thức: w_j ∝ (1/D_j)^(1/(β−1)), Σw = 1, β = 2 (kiểu
    Keller & Klawonn 2000) — đối chứng cho cách 7; khác WFCM trọng số theo cụm của bài 06 (thuật toán C)."""
    t0 = perf_counter()
    mo_hinh = FCM(c, m=m)
    d = X.shape[1]
    w = np.full(d, 1.0 / d)
    V = mo_hinh.init_centroids_kmeans_plusplus(X, seed=rng)
    n_vong = 0
    for n_vong in range(1, max_vong + 1):
        s = w ** (beta / 2)
        U = mo_hinh._cost_to_membership(cdist_sq_euclidean(X * s, V * s))
        V_moi = tam_tu_U(X, U, m)
        Dj = np.einsum("ik,ikj->j", U**m, (X[:, None, :] - V_moi[None, :, :]) ** 2) + 1e-12
        w = (1.0 / Dj) ** (1.0 / (beta - 1))
        w /= w.sum()
        hoi_tu = np.abs(V_moi - V).max() < tol
        V = V_moi
        if hoi_tu:
            break
    s = w ** (beta / 2)
    U = mo_hinh._cost_to_membership(cdist_sq_euclidean(X * s, V * s))
    # trong_so theo thang của bai_toan_trong_so (khoảng cách nhân λ_j / mean λ): λ = w^β, để chuyển sang mẫu khác (E10)
    return KetQua(U=U, V=V, n_vong_fcm=n_vong, t_tinh_chinh=perf_counter() - t0, thong_tin={"trong_so": w**beta})


W_MIN, W_MAX = 0.05, 1.0


def he_so_trong_so(w: np.ndarray) -> np.ndarray:
    """Hệ số nhân từng cột: √(w / mean w), để tổng phương sai của dữ liệu z-score vẫn là D."""
    return np.sqrt(w / w.mean())


def bai_toan_trong_so(X, c, m, rng):
    """Bài toán tìm trọng số thuộc tính w ∈ [W_MIN; W_MAX]^D cho FCM, trả về (V0, fitness).

    Chấm w: FCM (trần VONG_FCM_TRONG vòng) trên X·he_so(w) từ bộ tâm k-means++ cố định V0; fitness = −Silhouette của
    phân hoạch cứng, tính trên KHÔNG GIAN GỐC với mẫu cố định N_MAU_SILHOUETTE điểm, cộng phạt khi cụm cứng nhỏ nhất
    < 5%. Nhãn vỡ nợ không đi vào fitness."""
    n = len(X)
    V0 = FCM(c, m=m).init_centroids_kmeans_plusplus(X, seed=rng)
    mau = rng.choice(n, min(n, N_MAU_SILHOUETTE), replace=False)
    D_mau = cdist_euclidean(X[mau], X[mau])
    np.fill_diagonal(D_mau, 0.0)

    def fitness(w):
        s = he_so_trong_so(w)
        U, _, _ = FCM(c, m=m, max_iter=VONG_FCM_TRONG).fit(X * s, init_v=V0 * s, with_u=False)
        nhan = U[mau].argmax(axis=1)
        sil = silhouette_score(D_mau, nhan, metric="precomputed") if len(np.unique(nhan)) > 1 else -1.0
        return -sil + phat_kich_thuoc(U)

    return V0, fitness


def phan_hoach_trong_so(X, m, V0, w) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Phân hoạch cuối: FCM tới hội tụ trên X·he_so(w) từ V0; tâm trả về ở không gian của X."""
    s = he_so_trong_so(w)
    U, _, n_vong, t = tinh_chinh_fcm(X * s, V0 * s, m)
    return U, tam_tu_U(X, U, m), n_vong, t


def cach7_pso_trong_so(X, c, m, rng, max_fe=None) -> KetQua:
    """Hạt là w ∈ [0,05; 1]^D, fitness của bai_toan_trong_so; hạt đầu là w = 1 (FCM thường). Phân hoạch cuối chạy
    FCM tới hội tụ với w tốt nhất."""
    t0 = perf_counter()
    d = X.shape[1]
    V0, fitness = bai_toan_trong_so(X, c, m, rng)
    bay = np.vstack([np.ones(d), rng.uniform(W_MIN, W_MAX, size=(N_HAT_TRONG_SO - 1, d))])
    t1 = perf_counter()
    kq = pso(bay, W_MIN, W_MAX, fitness, rng, max_fe=max_fe)
    t2 = perf_counter()
    U, V, n_vong, t = phan_hoach_trong_so(X, m, V0, kq.gbest)
    return KetQua(U=U, V=V, lich_su=kq.lich_su, n_fe=kq.n_fe, n_vong_fcm=n_vong,
                  t_khoi_tao=t1 - t0, t_tim_kiem=t2 - t1, t_tinh_chinh=t)


# ------------------------------------------------------------------ loại II: kernel FCM
def _kfcm_U(X, V, m, sigma2):
    d = np.fmax(2.0 * (1.0 - np.exp(-cdist_sq_euclidean(X, V) / sigma2)), 1e-12)
    nghich = d ** (-1.0 / (m - 1.0))
    return nghich / nghich.sum(axis=1, keepdims=True), d


def _kfcm_J(X, V, m, sigma2):
    U, d = _kfcm_U(X, V, m, sigma2)
    return float(np.sum((U**m) * d))


def _kfcm_hoi_tu(X, V, m, sigma2, max_vong=10000, tol=1e-5):
    n_vong = 0
    for n_vong in range(1, max_vong + 1):
        U, _ = _kfcm_U(X, V, m, sigma2)
        Kw = (U**m) * np.exp(-cdist_sq_euclidean(X, V) / sigma2)
        V_moi = (Kw.T @ X) / np.fmax(Kw.sum(axis=0)[:, None], 1e-12)
        hoi_tu = np.abs(V_moi - V).max() < tol
        V = V_moi
        if hoi_tu:
            break
    return _kfcm_U(X, V, m, sigma2)[0], V, n_vong


def kfcm(X, c, m, rng, max_fe=None) -> KetQua:
    """Kernel FCM nhân Gauss σ² = D (dữ liệu z-score có tổng phương sai D), khởi tạo k-means++."""
    t0 = perf_counter()
    U, V, n_vong = _kfcm_hoi_tu(X, FCM(c, m=m).init_centroids_kmeans_plusplus(X, seed=rng), m, float(X.shape[1]))
    return KetQua(U=U, V=V, n_vong_fcm=n_vong, t_tinh_chinh=perf_counter() - t0)


def cach8_pso_kfcm(X, c, m, rng, max_fe=None) -> KetQua:
    sigma2 = float(X.shape[1])
    t0 = perf_counter()
    bay = khoi_tao_tam(X, c, rng)
    lb, ub = _gioi_han(X)
    t1 = perf_counter()
    kq = pso(bay, np.broadcast_to(lb, bay.shape[1:]), np.broadcast_to(ub, bay.shape[1:]),
             lambda V: _kfcm_J(X, V, m, sigma2), rng, max_fe=max_fe)
    t2 = perf_counter()
    U, V, n_vong = _kfcm_hoi_tu(X, kq.gbest, m, sigma2)
    return KetQua(U=U, V=V, lich_su=kq.lich_su, n_fe=kq.n_fe, n_vong_fcm=n_vong, t_khoi_tao=t1 - t0,
                  t_tim_kiem=t2 - t1, t_tinh_chinh=perf_counter() - t2)
