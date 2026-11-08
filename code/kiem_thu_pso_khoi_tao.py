# -*- coding: utf-8 -*-
"""Kiểm thử PSO làm bộ khởi tạo tâm cho FCM (E5). Chạy: python3 kiem_thu_pso_khoi_tao.py"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

import chay_e5
import chay_e6
import chay_e7
from phan_tich import mcnemar_cap, wilson
import chay_thuc_nghiem as ctn
import tien_xu_ly as tx
from fcm import FCM
from pso_khoi_tao import OMEGA, PHIEN_BAN, bay_diem_ngau_nhien, fcm_diem_ngau_nhien, pso_khoi_tao_fcm


def _ba_cum(n=90, seed=0):
    rng = np.random.default_rng(seed)
    tam = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
    return np.concatenate([t + rng.normal(size=(n // 3, 2)) for t in tam])


def test_pso_goc_khong_co_quan_tinh():
    assert all(OMEGA["goc"](t, 30) == 1.0 for t in (1, 15, 30))


def test_chay_du_so_vong_va_dem_lan_cham():
    X = _ba_cum()
    for pb in PHIEN_BAN:
        kq = pso_khoi_tao_fcm(X, 3, 2.0, np.random.default_rng(1), phien_ban=pb, n_hat=7, so_vong=12)
        assert kq.thong_tin["so_vong"] == 12, pb
        assert kq.n_fe == 7 * 13, (pb, kq.n_fe)
        assert len(kq.lich_su) == 13, pb


def test_bay_chi_gom_diem_du_lieu():
    X = _ba_cum()
    bay = bay_diem_ngau_nhien(X, 3, 5, np.random.default_rng(2))
    assert bay.shape == (5, 3, 2)
    for hat in bay:
        assert all(any(np.allclose(v, x) for x in X) for v in hat)
        assert len({tuple(v) for v in hat}) == 3


def test_fcm_sau_pso_khong_lam_tang_J():
    X = _ba_cum()
    for pb in PHIEN_BAN:
        kq = pso_khoi_tao_fcm(X, 3, 2.0, np.random.default_rng(3), phien_ban=pb, n_hat=10, so_vong=20)
        j = FCM(3, m=2.0).objective_from_centroids(X, kq.V)
        assert j <= kq.thong_tin["j_pso"] * (1 + 1e-9), (pb, j, kq.thong_tin["j_pso"])


def test_cung_seed_cung_ket_qua():
    X = _ba_cum()
    a = pso_khoi_tao_fcm(X, 3, 2.0, np.random.default_rng(4), phien_ban="w", n_hat=10, so_vong=20)
    b = pso_khoi_tao_fcm(X, 3, 2.0, np.random.default_rng(4), phien_ban="w", n_hat=10, so_vong=20)
    assert np.allclose(a.V, b.V) and a.lich_su == b.lich_su


def test_phien_ban_sai_bao_loi():
    try:
        pso_khoi_tao_fcm(_ba_cum(), 3, 2.0, np.random.default_rng(0), phien_ban="khong_co")
    except ValueError:
        return
    raise AssertionError("phải báo lỗi phiên bản không hợp lệ")


def test_fcm_diem_ngau_nhien_ghi_J_ban_dau():
    X = _ba_cum()
    kq = fcm_diem_ngau_nhien(X, 3, 2.0, np.random.default_rng(5))
    assert kq.thong_tin["j_pso"] >= FCM(3, m=2.0).objective_from_centroids(X, kq.V) * (1 - 1e-9)


def test_trinh_chay_ghi_du_dong_va_chay_tiep():
    X = _ba_cum(60)
    y = (np.arange(len(X)) % 3 == 0).astype(int)
    ctn.BO_THEM["ba_cum"] = lambda: tx.BoDuLieu(X=X, y=y, ten_cot=["a", "b"], ten="ba_cum", X_goc=X)
    ctn.nap.cache_clear()
    pp = ["fcm", "fcm_kpp", "pso_goc_n10_t20", "pso_qpso_n10_t20"]
    with tempfile.TemporaryDirectory() as tmp:
        tv = chay_e5.TacVu("ba_cum", 3, 1000)
        dong = chay_e5.chay_tac_vu(tv, pp)
        chay_e5._ghi(tmp, tv, dong)
        d = pd.read_csv(os.path.join(tmp, "E5", "ba_cum_c3.csv"))
        assert len(d) == 4 and d["loi"].fillna("").eq("").all(), d[["phuong_phap", "loi"]]
        assert set(d.loc[d.phuong_phap == "pso_goc_n10_t20", ["n_hat", "so_vong"]].iloc[0]) == {10, 20}
        chay_e5._ghi(tmp, tv, dong)  # ghi lại cùng seed không nhân đôi dòng
        assert len(pd.read_csv(os.path.join(tmp, "E5", "ba_cum_c3.csv"))) == 4
    del ctn.BO_THEM["ba_cum"]


def test_e6_tom_tat_dem_muc_va_ti_le_dat():
    dong = []
    for seed in range(4):
        for kieu in chay_e6.KIEU:
            J = 100.0 if (kieu == "fcm" and seed < 2) else 90.0
            dong.append(dict(bo="b", m=2.0, c=3, seed=seed, kieu=kieu, J=J, sup=False, kc_tam_min=1.0, PC=0.5, auc=0.6,
                             vong_fcm=10, t=0.1))
    tt = chay_e6.tom_tat(pd.DataFrame(dong)).iloc[0]
    assert tt.so_muc == 2, tt
    assert tt.dat_fcm == 50 and tt.dat_fcm_kpp == 100 and tt.dat_pso_w == 100, tt
    assert tt.k_fcm == 2 and tt.n_fcm == 4 and tt.k_pso_w == 4, tt


def test_wilson_bien_va_doi_xung():
    lo, hi = wilson(30, 30)
    assert abs(hi - 100) < 1e-9 and 88 < lo < 89, (lo, hi)
    lo0, hi0 = wilson(0, 30)
    assert lo0 == 0 and abs(hi0 - (100 - lo)) < 1e-9, (lo0, hi0)


def test_mcnemar_cap_dem_bat_dong_theo_seed():
    a = pd.Series([1, 1, 0, 1], index=[10, 11, 12, 13])
    b = pd.Series([1, 0, 0, 0], index=[13, 11, 12, 10])
    chi_a, chi_b, p = mcnemar_cap(a, b)
    assert (chi_a, chi_b) == (2, 0) and abs(p - 0.5) < 1e-12, (chi_a, chi_b, p)
    assert mcnemar_cap(a, a)[2] == 1.0


def test_e7_vong_cai_thien_cuoi():
    ls = np.array([[5.0, 4.0, 4.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0, 5.0], [5.0, 4.0, 4.0, 4.0, 4.0]])
    assert list(chay_e7.vong_cai_thien_cuoi(ls)) == [3, 0, 1]


if __name__ == "__main__":
    tests = [(ten, fn) for ten, fn in globals().items() if ten.startswith("test_")]
    loi = 0
    for ten, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            loi += 1
            print(f"KHÔNG ĐẠT {ten}: {exc!r}")
    print(f"{len(tests) - loi}/{len(tests)} phép thử đạt")
    raise SystemExit(1 if loi else 0)
