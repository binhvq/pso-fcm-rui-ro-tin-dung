# -*- coding: utf-8 -*-
"""Kiểm thử hàm đánh giá một lượt. Chạy: python3 kiem_thu_danh_gia.py"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from danh_gia import COT, danh_gia, mau_silhouette
from fcm import FCM
from giao_dien import KetQua, fcm_kpp


def _iris():
    return load_iris().data.astype(np.float64)


def _ket_qua_one_hot(nhan, c, X):
    U = np.eye(c)[nhan]
    V = np.stack([X[nhan == k].mean(axis=0) if np.any(nhan == k) else X.mean(axis=0) + k for k in range(c)])
    return KetQua(U=U, V=V)


def test_auc_bang_1_khi_mot_cum_chua_dung_nhan_1():
    X = _iris()
    nhan = np.repeat([0, 1, 2], 50)
    y = (nhan == 1).astype(int)
    d = danh_gia(X, y, _ket_qua_one_hot(nhan, 3, X), mau=mau_silhouette(len(X)))
    assert d["auc_rui_ro"] == 1.0 and d["chenh_ti_le"] == 1.0, d


def test_auc_bang_0_5_khi_do_thuoc_hang():
    X = _iris()
    y = np.repeat([0, 1, 0], 50)
    kq = KetQua(U=np.full((len(X), 3), 1 / 3), V=X[[0, 60, 120]])
    assert danh_gia(X, y, kq, mau=mau_silhouette(len(X)))["auc_rui_ro"] == 0.5


def test_auc_cung_bang_auc_rui_ro_khi_one_hot_hai_cum():
    X = _iris()
    rng = np.random.default_rng(0)
    nhan = rng.integers(0, 2, len(X))
    y = (rng.random(len(X)) < np.where(nhan == 1, 0.6, 0.3)).astype(int)
    d = danh_gia(X, y, _ket_qua_one_hot(nhan, 2, X), mau=mau_silhouette(len(X)))
    assert np.isclose(d["auc_cung"], d["auc_rui_ro"]), d


def test_auc_cung_khong_duoi_0_5_du_auc_rui_ro_duoi_0_5():
    X = _iris()
    nhan = np.repeat([0, 1], 75)
    y = np.r_[np.zeros(60), np.ones(15), np.zeros(55), np.ones(20)].astype(int)  # cụm 1 rủi ro hơn
    U = np.where(np.eye(2)[nhan] == 1, 0.55, 0.45)
    U[nhan == 1, 1] = np.r_[np.full(55, 0.9), np.full(20, 0.51)]  # độ thuộc ngược chiều rủi ro trong cụm
    U[:, 0] = 1 - U[:, 1]
    d = danh_gia(X, y, KetQua(U=U, V=X[[0, 100]]), mau=mau_silhouette(len(X)))
    assert d["auc_rui_ro"] < 0.5 <= d["auc_cung"], d


def test_cum_rong_va_nho_nhat():
    X = _iris()
    nhan = np.repeat([0, 2], 75)
    d = danh_gia(X, np.repeat([0, 1], 75), _ket_qua_one_hot(nhan, 3, X), mau=mau_silhouette(len(X)))
    assert d["so_cum_rong"] == 1 and d["cum_nho_nhat"] == 0, d


def test_J_m_khop_fcm():
    X = _iris()
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(0))
    mo_hinh = FCM(3)
    U_toi_uu = mo_hinh._update_membership(X, kq.V)
    d = danh_gia(X, np.zeros(len(X), dtype=int) + (np.arange(len(X)) < 50), kq, mau=mau_silhouette(len(X)))
    assert np.isclose(d["J_m"], mo_hinh.compute_objective_j(X, U_toi_uu, kq.V), rtol=1e-12)


def test_khoa_va_thu_tu_co_dinh():
    X = _iris()
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(0))
    assert list(danh_gia(X, (np.arange(len(X)) < 50).astype(int), kq, mau=mau_silhouette(len(X)))) == COT


def test_silhouette_dung_mau_co_dinh():
    X = _iris()
    y = (np.arange(len(X)) < 50).astype(int)
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(0))
    assert np.array_equal(mau_silhouette(len(X)), mau_silhouette(len(X)))
    assert danh_gia(X, y, kq, mau=mau_silhouette(len(X)))["silhouette"] == \
        danh_gia(X, y, kq, mau=mau_silhouette(len(X)))["silhouette"]
    assert len(mau_silhouette(10_000)) == 3000 and len(np.unique(mau_silhouette(10_000))) == 3000


def test_tam_trung_nhau_cho_XB_vo_cung_khong_loi():
    X = _iris()
    nhan = np.repeat([0, 1, 2], 50)
    kq = _ket_qua_one_hot(nhan, 3, X)
    kq.V[2] = kq.V[1]
    d = danh_gia(X, (nhan == 1).astype(int), kq, mau=mau_silhouette(len(X)))
    assert d["XB"] == np.inf and np.isfinite(d["auc_rui_ro"])


def test_goi_pso_lay_tu_thong_tin():
    X = _iris()
    y = (np.arange(len(X)) < 50).astype(int)
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(0))
    assert danh_gia(X, y, kq, mau=mau_silhouette(len(X)))["goi_pso"] == ""
    kq.thong_tin["goi_pso"] = True
    assert danh_gia(X, y, kq, mau=mau_silhouette(len(X)))["goi_pso"] is True


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ĐẠT   {name}")
        except Exception as exc:
            failed += 1
            print(f"LỖI   {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} phép thử đạt")
    raise SystemExit(1 if failed else 0)
