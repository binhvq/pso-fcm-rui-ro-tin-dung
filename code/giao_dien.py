# -*- coding: utf-8 -*-
"""Giao diện chung cho mọi thuật toán phân cụm trong thực nghiệm.

Mỗi thuật toán (K-means, FCM, A, B, C, đề xuất) là một hàm cùng chữ ký

    fit(X, c, m, rng, max_fe) -> KetQua

- X: ma trận dữ liệu (n, D) đã tiền xử lý; hàm không được sửa X.
- c: số cụm; m: hệ số mờ (K-means bỏ qua).
- rng: np.random.Generator — nguồn ngẫu nhiên duy nhất, để cùng seed cho cùng kết quả.
- max_fe: số lần chấm fitness tối đa của pha tìm kiếm metaheuristic; None = không giới hạn.
  Thuật toán không có pha tìm kiếm (FCM, K-means) bỏ qua và báo n_fe = 0.

Nhãn vỡ nợ không nằm trong chữ ký: không thuật toán nào được nhìn nhãn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans

from fcm import FCM


@dataclass
class KetQua:
    U: np.ndarray  # (n, c) độ thuộc; phân cụm cứng dùng one-hot
    V: np.ndarray  # (c, D) tâm cụm trong không gian của X
    lich_su: list[float] = field(default_factory=list)  # fitness tốt nhất sau mỗi vòng tìm kiếm (không tăng)
    n_fe: int = 0  # số lần chấm fitness của pha tìm kiếm
    n_vong_fcm: int = 0  # số vòng FCM của pha tinh chỉnh / phân hoạch cuối
    t_khoi_tao: float = 0.0  # giây, đo bằng perf_counter
    t_tim_kiem: float = 0.0
    t_tinh_chinh: float = 0.0
    thong_tin: dict = field(default_factory=dict)  # thông tin riêng của từng thuật toán (ví dụ kết quả chẩn đoán)

    @property
    def t_tong(self) -> float:
        return self.t_khoi_tao + self.t_tim_kiem + self.t_tinh_chinh

    @property
    def nhan(self) -> np.ndarray:
        return np.argmax(self.U, axis=1)


HamFit = Callable[[np.ndarray, int, float, np.random.Generator, "int | None"], KetQua]


def tinh_chinh_fcm(X: np.ndarray, V0: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Chạy FCM tới hội tụ (ε và trần vòng mặc định của FCM) từ bộ tâm V0.

    Dùng chung cho pha tinh chỉnh của A, B, C và phân hoạch cuối của đề xuất, để mọi phương pháp
    kết thúc ở cùng tiêu chuẩn hội tụ với FCM đối chứng.
    """
    t0 = perf_counter()
    model = FCM(V0.shape[0], m=m)
    U, V, n_iter = model.fit(X, init_v=V0, with_u=False)
    return U, V, n_iter, perf_counter() - t0


def fcm_mot_lan(X, c, m, rng, max_fe=None) -> KetQua:
    """FCM khởi tạo U ngẫu nhiên, chạy một lần tới hội tụ."""
    t0 = perf_counter()
    model = FCM(c, m=m)
    U, V, n_iter = model.fit(X, seed=rng)
    return KetQua(U=U, V=V, n_vong_fcm=n_iter, t_tinh_chinh=perf_counter() - t0)


def fcm_kpp(X, c, m, rng, max_fe=None) -> KetQua:
    """FCM khởi tạo tâm k-means++ (một lần), chạy tới hội tụ — cùng kiểu khởi tạo với các phương pháp có PSO."""
    t0 = perf_counter()
    model = FCM(c, m=m)
    V0 = model.init_centroids_kmeans_plusplus(X, seed=rng)
    t1 = perf_counter()
    U, V, n_iter = model.fit(X, init_v=V0, with_u=False)
    return KetQua(U=U, V=V, n_vong_fcm=n_iter, t_khoi_tao=t1 - t0, t_tinh_chinh=perf_counter() - t1)


def kmeans(X, c, m, rng, max_fe=None) -> KetQua:
    """K-means (k-means++, một lần khởi tạo); U là one-hot của nhãn cứng."""
    t0 = perf_counter()
    km = KMeans(n_clusters=c, init="k-means++", n_init=1, random_state=int(rng.integers(2**31 - 1)))
    nhan = km.fit_predict(X)
    U = np.eye(c)[nhan]
    return KetQua(U=U, V=km.cluster_centers_.copy(), t_tinh_chinh=perf_counter() - t0)


def kiem_tra_giao_dien(fit: HamFit, X: np.ndarray, c: int, m: float = 2.0, seed: int = 0,
                       max_fe: int | None = 200) -> KetQua:
    """Kiểm tra một hàm fit có tuân giao diện chung; báo AssertionError kèm lý do nếu không.

    Điều kiện: đúng kiểu và kích thước; U hợp lệ (không âm, tổng mỗi hàng bằng 1); V hữu hạn; không sửa X;
    n_fe ≤ max_fe; thời gian không âm; lịch sử fitness không tăng; cùng seed cho cùng kết quả.
    """
    X_goc = X.copy()
    kq = fit(X, c, m, np.random.default_rng(seed), max_fe)
    n, d = X.shape
    assert isinstance(kq, KetQua), f"phải trả về KetQua, nhận {type(kq).__name__}"
    assert np.array_equal(X, X_goc), "fit đã sửa dữ liệu đầu vào X"
    assert kq.U.shape == (n, c), f"U phải có kích thước {(n, c)}, nhận {kq.U.shape}"
    assert kq.V.shape == (c, d), f"V phải có kích thước {(c, d)}, nhận {kq.V.shape}"
    assert np.all(np.isfinite(kq.U)) and np.all(np.isfinite(kq.V)), "U hoặc V có NaN/inf"
    assert np.all(kq.U >= -1e-12), "U có phần tử âm"
    assert np.allclose(kq.U.sum(axis=1), 1.0, atol=1e-6), "tổng độ thuộc mỗi điểm phải bằng 1"
    assert kq.n_fe >= 0 and kq.n_vong_fcm >= 0, "n_fe, n_vong_fcm phải không âm"
    if max_fe is not None:
        assert kq.n_fe <= max_fe, f"vượt ngân sách: n_fe = {kq.n_fe} > max_fe = {max_fe}"
    assert min(kq.t_khoi_tao, kq.t_tim_kiem, kq.t_tinh_chinh) >= 0, "thời gian âm"
    ls = np.asarray(kq.lich_su, dtype=float)
    assert np.all(np.isfinite(ls)), "lịch sử fitness có NaN/inf"
    assert np.all(np.diff(ls) <= 1e-12 * np.maximum(1.0, np.abs(ls[:-1]))), \
        "lịch sử fitness tốt nhất phải không tăng"
    kq2 = fit(X, c, m, np.random.default_rng(seed), max_fe)
    assert np.allclose(kq.U, kq2.U) and np.allclose(kq.V, kq2.V) and kq.n_fe == kq2.n_fe, \
        "cùng seed phải cho cùng kết quả (chỉ dùng rng được truyền vào)"
    return kq
