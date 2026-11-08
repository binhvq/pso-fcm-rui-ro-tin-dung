# -*- coding: utf-8 -*-
"""Kiểm thử PSO cơ sở, các cách kết hợp và đề xuất. Chạy: python3 kiem_thu_cach_ket_hop.py"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

import cach_ket_hop as ck
from de_xuat import de_xuat_ep_pso, pso_fcm_chan_doan
from fcm import FCM
from giao_dien import kiem_tra_giao_dien
from pso_co_so import N_HAT, omega_tuyen_tinh, pso
from thuat_toan_a import OMEGA, a_L, a_LOG, a_S
from thuat_toan_b import qpso_fcm
from thuat_toan_c import _khoi_tao_da_dang, _wfcm_cham, msgwo_wfcm

PHUONG_PHAP = {
    "fcm_lap_lai": ck.fcm_lap_lai,
    "cach2_fcm_gieo": ck.cach2_fcm_gieo,
    "cach3_xen_ke": ck.cach3_xen_ke,
    "a_L": a_L,
    "a_S": a_S,
    "a_LOG": a_LOG,
    "b_qpso_fcm": qpso_fcm,
    "c_msgwo_wfcm": msgwo_wfcm,
    "cach5_pso_U": ck.cach5_pso_U,
    "wfcm_giai_tich": ck.wfcm_giai_tich,
    "cach7_pso_trong_so": ck.cach7_pso_trong_so,
    "kfcm": ck.kfcm,
    "cach8_pso_kfcm": ck.cach8_pso_kfcm,
    "de_xuat": pso_fcm_chan_doan,
    "de_xuat_ep_pso": de_xuat_ep_pso,
}


def _iris():
    return load_iris().data.astype(np.float64)


def test_pso_toi_thieu_ham_cau():
    rng = np.random.default_rng(0)
    kq = pso(rng.uniform(-5, 5, (N_HAT, 4)), -5.0, 5.0, lambda x: float(np.sum(x**2)), rng)
    assert kq.gbest_f < 1e-3, kq.gbest_f
    assert np.all(np.diff(kq.lich_su) <= 0) and len(kq.lich_su) == kq.n_vong + 1


def test_pso_ton_trong_ngan_sach():
    rng = np.random.default_rng(0)
    kq = pso(rng.uniform(-5, 5, (N_HAT, 4)), -5.0, 5.0, lambda x: float(np.sum(x**2)), rng, max_fe=95)
    assert kq.n_fe <= 95 and kq.n_fe % N_HAT == 0


def test_omega_tuyen_tinh_dau_cuoi():
    assert np.isclose(omega_tuyen_tinh(1, 150), 0.9) and np.isclose(omega_tuyen_tinh(150, 150), 0.4)


def test_moi_phuong_phap_dat_giao_dien():
    X = _iris()
    for ten, fit in PHUONG_PHAP.items():
        try:
            kiem_tra_giao_dien(fit, X, 3, max_fe=400)
        except AssertionError as exc:
            raise AssertionError(f"{ten}: {exc}") from exc


def test_loai_I_cung_nghiem_fcm_khi_mot_cuc_tieu():
    X = _iris()
    J_fcm = FCM(3).objective_from_centroids(X, ck.fcm_lap_lai(X, 3, 2.0, np.random.default_rng(0)).V)
    for fit in (ck.cach2_fcm_gieo, ck.cach3_xen_ke):
        V = fit(X, 3, 2.0, np.random.default_rng(1), 400).V
        assert np.isclose(FCM(3).objective_from_centroids(X, V), J_fcm, rtol=1e-6), fit.__name__


def test_cach7_trong_so_trong_mien_va_phan_hoach_hoi_tu():
    X = _iris()
    kq = ck.cach7_pso_trong_so(X, 3, 2.0, np.random.default_rng(0), max_fe=60)
    assert kq.n_fe <= 60 and kq.n_vong_fcm < FCM(3)._max_iter


def test_de_xuat_mot_cuc_tieu_khong_goi_pso():
    kq = pso_fcm_chan_doan(_iris(), 3, 2.0, np.random.default_rng(0))
    assert kq.thong_tin["goi_pso"] is False and kq.n_fe == 0 and kq.t_tim_kiem == 0


def test_de_xuat_nhieu_cuc_tieu_goi_pso_va_khong_te_hon_fcm():
    X = _iris()
    kq = pso_fcm_chan_doan(X, 6, 2.0, np.random.default_rng(0), max_fe=400)
    assert kq.thong_tin["goi_pso"] is True and kq.n_fe > 0
    J = FCM(6).objective_from_centroids(X, kq.V)
    assert J <= min(kq.thong_tin["J_chan_doan"]) * (1 + 1e-9), "không được tệ hơn nghiệm FCM tốt nhất khi chẩn đoán"


def test_de_xuat_ep_pso_luon_goi_pso():
    X = _iris()
    kq = de_xuat_ep_pso(X, 3, 2.0, np.random.default_rng(0), max_fe=400)
    assert kq.thong_tin["goi_pso"] is True and kq.thong_tin["ep_pso"] is True and kq.n_fe > 0
    J = FCM(3).objective_from_centroids(X, kq.V)
    assert J <= min(kq.thong_tin["J_chan_doan"]) * (1 + 1e-9)


def test_phat_kich_thuoc():
    U = np.eye(2)[np.array([0] * 97 + [1] * 3)]
    assert np.isclose(ck.phat_kich_thuoc(U), (0.05 - 0.03) / 0.05)
    assert ck.phat_kich_thuoc(np.eye(2)[np.array([0, 1] * 50)]) == 0.0


def test_omega_ba_kieu_dung_cong_thuc_bai_02():
    assert np.isclose(OMEGA["L"](150, 150), 0.3) and np.isclose(OMEGA["S"](75, 150), 1.2 - 0.9 * 0.25)
    assert np.isclose(OMEGA["LOG"](0, 150), np.log(2.98) ** -1.51) and 0.6 < OMEGA["LOG"](150, 150) < 0.62


def test_b_cung_nghiem_fcm_khi_mot_cuc_tieu():
    X = _iris()
    J_fcm = FCM(3).objective_from_centroids(X, ck.fcm_lap_lai(X, 3, 2.0, np.random.default_rng(0)).V)
    V = qpso_fcm(X, 3, 2.0, np.random.default_rng(1), 400).V
    assert np.isclose(FCM(3).objective_from_centroids(X, V), J_fcm, rtol=1e-6)


def test_c_trong_so_theo_cum_tong_bang_1_va_khoi_tao_phu_doan():
    X = _iris()
    V = X[:3]
    _, U, W = _wfcm_cham(X, V, np.full((3, 4), 0.25), 2.0)
    assert np.allclose(W.sum(axis=1), 1.0) and np.allclose(U.sum(axis=1), 1.0)
    lb, ub = np.zeros((2, 3)), np.ones((2, 3))
    P = _khoi_tao_da_dang(20, lb, ub, np.random.default_rng(0))
    doan = np.floor(P[10:] * 10).astype(int)  # nửa sau: mỗi chiều phủ đủ 10 đoạn, mỗi đoạn một lần
    assert all(sorted(doan[:, i, j]) == list(range(10)) for i in range(2) for j in range(3))


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
