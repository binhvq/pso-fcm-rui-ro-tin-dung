# -*- coding: utf-8 -*-
"""Kiểm thử tiền xử lý dữ liệu. Chạy: python3 kiem_thu_tien_xu_ly.py"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import tien_xu_ly as tx

KICH_THUOC = {  # (n, D) sau tiền xử lý, tỉ lệ nhãn 1
    "viet_nam": (3277, 21, 0.133),
    "taiwan_kiem_chung": (5000, 23, 0.2212),
    "taiwan_kiem_chung_khong_log": (5000, 23, 0.2212),
    "taiwan_thiet_ke": (5000, 23, 0.2212),
    "south_german": (1000, 20, 0.30),
    "german_so": (1000, 24, 0.30),
    "australian": (690, 14, 0.555),
}


def test_kich_thuoc_va_ti_le_nhan():
    for ten, (n, d, ti_le) in KICH_THUOC.items():
        bo = tx.nap(ten)
        assert bo.X.shape == (n, d), (ten, bo.X.shape)
        assert len(bo.ten_cot) == d and len(bo.y) == n, ten
        assert abs(bo.y.mean() - ti_le) < 0.002, (ten, bo.y.mean())


def test_khong_thieu_va_da_chuan_hoa():
    for ten in tx.DANH_SACH:
        X = tx.nap(ten).X
        assert np.all(np.isfinite(X)), ten
        assert np.allclose(X.mean(axis=0), 0, atol=1e-8) and np.allclose(X.std(axis=0), 1, atol=1e-8), ten


def _goc(thu_muc, mau):
    return glob.glob(os.path.join(tx.RAW, thu_muc, mau))[0]


def test_viet_nam_bo_cot_trung_va_dien_so_0_gia():
    bo = tx.nap("viet_nam")
    assert not {"X3", "X15", "X18", "X21"} & set(bo.ten_cot)
    v = pd.read_csv(_goc("vn-financial-distress-listed-firms", "*.csv"), encoding="latin-1")
    for col in ("X6", "X7", "X12"):
        bang_0 = (v[col] == 0).values
        assert bang_0.sum() > 1000, col
        gia_tri = np.unique(bo.X[bang_0, bo.ten_cot.index(col)])
        assert gia_tri.size == 1, f"{col}: các ô thiếu giả phải được điền cùng một trung vị"
        trung_vi = np.median(v.loc[~bang_0, col])
        assert np.isclose(np.unique(bo.X[(v[col] == trung_vi).values, bo.ten_cot.index(col)])[0], gia_tri[0]), col


def test_hai_mau_taiwan_khong_giao_nhau_va_tai_lap():
    y = pd.read_excel(_goc("taiwan-default", "*.xls"), header=1).iloc[:, -1].values
    tk = tx.mau_phan_tang(y, tx.N_MAU, np.random.default_rng(0))
    kc = tx.mau_phan_tang(y, tx.N_MAU, np.random.default_rng(1), trong=np.setdiff1d(np.arange(len(y)), tk))
    assert len(np.unique(tk)) == len(np.unique(kc)) == tx.N_MAU
    assert len(np.intersect1d(tk, kc)) == 0, "mẫu kiểm chứng không được trùng mẫu thiết kế"
    a, b = tx.nap("taiwan_kiem_chung"), tx.nap("taiwan_kiem_chung")
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y), "nạp lại phải cho cùng mẫu"


def test_log_co_dau():
    x = np.array([[-9.0, 0.0, 9.0]])
    assert np.allclose(tx.signed_log(x), [[-np.log(10), 0.0, np.log(10)]])


KHONG_LOG = {"taiwan_kiem_chung_khong_log", "south_german", "german_so", "australian"}
HOI_QUY_X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mau_kiem_thu", "hoi-quy-X.npz")  # tao_mau_hoi_quy.py


def _zscore(A):
    A = A[:, A.std(axis=0) > 0]
    return (A - A.mean(axis=0)) / A.std(axis=0)


def test_X_goc_cung_kich_thuoc_va_suy_ra_X():
    for ten in tx.DANH_SACH:
        bo = tx.nap(ten)
        assert bo.X_goc.shape == bo.X.shape, (ten, bo.X_goc.shape, bo.X.shape)
        A = bo.X_goc if ten in KHONG_LOG else tx.signed_log(bo.X_goc)
        assert np.allclose(_zscore(A), bo.X, atol=1e-10), ten


def test_australian_khong_log():
    bo = tx.nap("australian")
    assert np.allclose(bo.X, _zscore(bo.X_goc), atol=1e-12)


def test_hoi_quy_X_khong_doi():
    cu = np.load(HOI_QUY_X)
    for ten in tx.DANH_SACH:
        if ten == "australian":
            continue
        assert np.allclose(tx.nap(ten).X, cu[ten], rtol=0, atol=1e-12), ten
    assert not np.allclose(tx.nap("australian").X, cu["australian"]), "Australian phải đổi vì bỏ log"


def test_bo_khong_ton_tai_bao_loi():
    try:
        tx.nap("khong_co")
    except KeyError:
        return
    raise AssertionError("phải báo KeyError")



def test_viet_nam_nua_tach_theo_doanh_nghiep():
    a, b, du = tx.viet_nam_nua(0), tx.viet_nam_nua(1), tx.nap("viet_nam")
    assert len(a.y) + len(b.y) == len(du.y) and a.ten_cot == b.ten_cot == du.ten_cot
    _, _, cty = tx._viet_nam_bang(tx.RAW)
    ma = np.random.default_rng(0).permutation(np.unique(cty))
    assert len(np.intersect1d(ma[: len(ma) // 2], ma[len(ma) // 2:])) == 0
    for bo in (a, b):
        assert np.allclose(bo.X.mean(axis=0), 0, atol=1e-10) and np.allclose(bo.X.std(axis=0), 1, atol=1e-10)

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
