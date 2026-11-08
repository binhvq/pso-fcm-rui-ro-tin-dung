# -*- coding: utf-8 -*-
"""Kiểm thử giao diện chung fit(X, c, m, rng, max_fe). Chạy: python3 kiem_thu_giao_dien.py"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from fcm import FCM
from giao_dien import fcm_kpp, fcm_mot_lan, kiem_tra_giao_dien, kmeans, tinh_chinh_fcm


def _iris():
    return load_iris().data.astype(np.float64)


def _phai_bao_loi(fit, thong_diep):
    try:
        kiem_tra_giao_dien(fit, _iris(), 3)
    except AssertionError as exc:
        assert thong_diep in str(exc), f"bắt sai lỗi: {exc}"
        return
    raise AssertionError(f"kiểm tra không phát hiện lỗi '{thong_diep}'")


def test_fcm_dat_giao_dien():
    kq = kiem_tra_giao_dien(fcm_mot_lan, _iris(), 3)
    assert kq.n_fe == 0 and kq.n_vong_fcm > 0 and kq.t_tong > 0


def test_kmeans_dat_giao_dien():
    kq = kiem_tra_giao_dien(kmeans, _iris(), 3)
    assert set(np.unique(kq.U)) <= {0.0, 1.0}


def test_fcm_kpp_dat_giao_dien():
    kq = kiem_tra_giao_dien(fcm_kpp, _iris(), 3)
    assert kq.n_fe == 0 and kq.n_vong_fcm > 0 and kq.t_tong > 0


def test_fcm_kpp_khop_fcm_tu_tam_kpp():
    X = _iris()
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(7))
    V0 = FCM(3).init_centroids_kmeans_plusplus(X, seed=np.random.default_rng(7))
    U, V, n_iter = FCM(3).fit(X, init_v=V0, with_u=False)
    assert np.array_equal(kq.U, U) and np.array_equal(kq.V, V) and kq.n_vong_fcm == n_iter


def test_seed_khac_cho_khoi_tao_khac():
    X = _iris()
    a = fcm_mot_lan(X, 3, 2.0, np.random.default_rng(1))
    b = fcm_mot_lan(X, 3, 2.0, np.random.default_rng(2))
    assert a.n_vong_fcm != b.n_vong_fcm or not np.array_equal(a.U, b.U)


def test_tinh_chinh_fcm_chay_toi_hoi_tu():
    X = _iris()
    V0 = FCM(3).init_centroids_kmeans_plusplus(X, seed=0)
    U, V, n_iter, t = tinh_chinh_fcm(X, V0, 2.0)
    U_ref, V_ref, n_ref = FCM(3).fit(X, init_v=V0, with_u=False)
    assert np.allclose(U, U_ref) and np.allclose(V, V_ref) and n_iter == n_ref
    assert n_iter < FCM(3)._max_iter, "phải dừng vì hội tụ, không phải vì chạm trần vòng"
    assert FCM(3).objective_from_centroids(X, V) <= FCM(3).objective_from_centroids(X, V0) + 1e-9 and t > 0


def test_bat_loi_vuot_ngan_sach():
    def sai(X, c, m, rng, max_fe):
        kq = fcm_mot_lan(X, c, m, rng)
        kq.n_fe = (max_fe or 0) + 1
        return kq

    _phai_bao_loi(sai, "vượt ngân sách")


def test_bat_loi_do_thuoc_khong_chuan_hoa():
    def sai(X, c, m, rng, max_fe):
        kq = fcm_mot_lan(X, c, m, rng)
        kq.U = kq.U * 2
        return kq

    _phai_bao_loi(sai, "tổng độ thuộc")


def test_bat_loi_dung_rng_toan_cuc():
    def sai(X, c, m, rng, max_fe):
        return fcm_mot_lan(X, c, m, np.random.default_rng())

    _phai_bao_loi(sai, "cùng seed")


def test_bat_loi_sua_du_lieu():
    def sai(X, c, m, rng, max_fe):
        X -= X.mean(axis=0)
        return fcm_mot_lan(X, c, m, rng)

    _phai_bao_loi(sai, "sửa dữ liệu")


def test_bat_loi_lich_su_tang():
    def sai(X, c, m, rng, max_fe):
        kq = fcm_mot_lan(X, c, m, rng)
        kq.lich_su = [3.0, 2.0, 2.5]
        return kq

    _phai_bao_loi(sai, "không tăng")


def test_bat_loi_sai_kieu_tra_ve():
    def sai(X, c, m, rng, max_fe):
        kq = fcm_mot_lan(X, c, m, rng)
        return kq.U, kq.V

    _phai_bao_loi(sai, "KetQua")


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
