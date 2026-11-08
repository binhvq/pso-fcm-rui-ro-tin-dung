# -*- coding: utf-8 -*-
"""Kiểm thử trình chạy thực nghiệm. Chạy: python3 kiem_thu_trinh_chay.py

Phép thử chỉ ghi vào thư mục tạm; dùng Iris khi không cần dữ liệu thật để giữ thời gian ngắn.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import chay_thuc_nghiem as ctn
import tien_xu_ly as tx
from danh_gia import COT


def _iris() -> tx.BoDuLieu:
    d = load_iris()
    X = (d.data - d.data.mean(axis=0)) / d.data.std(axis=0)
    return tx.BoDuLieu("iris", X, (d.target == 2).astype(int), list(d.feature_names), d.data.copy())


ctn.BO_THEM["iris"] = _iris
TAC_VU_IRIS = ctn.TacVu("E2", "iris", 3, 1000)


def _bo_thoi_gian(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("t_")]).reset_index(drop=True)


def test_dang_ky_du_phuong_phap():
    assert len(ctn.PHUONG_PHAP["E1"]) == 17 and len(set(ctn.PHUONG_PHAP["E1"])) == 17
    assert len(ctn.PHUONG_PHAP["E2"]) == 10 and len(set(ctn.PHUONG_PHAP["E2"])) == 10
    for ds in ctn.PHUONG_PHAP.values():
        for ten in ds:
            assert callable(ctn.HAM[ten]), ten
    assert {"fcm_kpp", "de_xuat"} <= set(ctn.PHUONG_PHAP["E1"]) and "de_xuat_ep_pso" in ctn.PHUONG_PHAP["E2"]


def test_so_tac_vu():
    assert len(ctn.danh_sach_tac_vu("E1")) == 2 * 3 * 30
    assert len(ctn.danh_sach_tac_vu("E1", nhanh=True)) == 2 * 3 * 3
    assert len(ctn.danh_sach_tac_vu("E2")) == 2 * 30
    assert len(ctn.danh_sach_tac_vu("E3")) == 6 * 3 + 2 * 5
    assert len(ctn.danh_sach_tac_vu("E4")) == 2 * 4  # fcm_kpp, kmeans, cach7_pso_trong_so, pso_trong_so


def test_mot_tac_vu_ghi_dung_so_dong():
    with tempfile.TemporaryDirectory() as tmp:
        ctn.chay_va_ghi([ctn.TacVu("E2", "taiwan_kiem_chung_khong_log", 3, 1000)], tmp, so_tien_trinh=1)
        df = pd.read_csv(os.path.join(tmp, "E2", "taiwan_kiem_chung_khong_log_c3.csv"))
        assert len(df) == 10 and set(df["phuong_phap"]) == set(ctn.PHUONG_PHAP["E2"])
        assert set(ctn.KHOA + COT + ["loi"]) <= set(df.columns)
        assert df["loi"].isna().all(), df["loi"].dropna().tolist()
        assert os.path.exists(os.path.join(tmp, "hoi-tu", "E2_taiwan_kiem_chung_khong_log_c3_a_LOG.json"))


def test_tai_lap_mot_tac_vu():
    a = pd.DataFrame(ctn.chay_tac_vu(TAC_VU_IRIS)[0])
    b = pd.DataFrame(ctn.chay_tac_vu(TAC_VU_IRIS)[0])
    pd.testing.assert_frame_equal(_bo_thoi_gian(a), _bo_thoi_gian(b))


def test_chay_tiep_bo_qua_khoa_da_co():
    with tempfile.TemporaryDirectory() as tmp:
        ctn.chay_va_ghi([TAC_VU_IRIS], tmp, so_tien_trinh=1)
        ctn.chay_va_ghi([TAC_VU_IRIS], tmp, so_tien_trinh=1)
        assert len(pd.read_csv(os.path.join(tmp, "E2", "iris_c3.csv"))) == 10


def test_chay_tiep_chay_lai_dong_loi():
    with tempfile.TemporaryDirectory() as tmp:
        ctn.chay_va_ghi([TAC_VU_IRIS], tmp, so_tien_trinh=1)
        tep = os.path.join(tmp, "E2", "iris_c3.csv")
        df = pd.read_csv(tep)
        df.loc[df["phuong_phap"] == "fcm", "loi"] = "RuntimeError('giả')"
        df.to_csv(tep, index=False)
        ctn.chay_va_ghi([TAC_VU_IRIS], tmp, so_tien_trinh=1)
        df = pd.read_csv(tep)
        assert len(df) == 10 and df["loi"].isna().all(), df[["phuong_phap", "loi"]]


def test_doc_lap_thu_tu_phuong_phap():
    xuoi = pd.DataFrame(ctn.chay_tac_vu(TAC_VU_IRIS)[0]).set_index("phuong_phap")
    nguoc = pd.DataFrame(ctn.chay_tac_vu(TAC_VU_IRIS, phuong_phap=ctn.PHUONG_PHAP["E2"][::-1])[0])
    nguoc = nguoc.set_index("phuong_phap").loc[xuoi.index]
    pd.testing.assert_frame_equal(_bo_thoi_gian(xuoi), _bo_thoi_gian(nguoc))


def test_e3_mot_bo():
    dong = ctn.chay_e3(ctn.TacVu("E3", "south_german", 2, 5))
    df = pd.DataFrame(dong)
    assert len(df) == 10 and sorted(df["kieu"].unique()) == ["kmeans_pp", "u_ngau_nhien"]
    assert (df.loc[df["kieu"] == "u_ngau_nhien", "sup"] == 1).all(), df
    assert np.isfinite(df["J_m"]).all()


def test_e4_profile():
    bo = tx.nap("viet_nam")
    df = pd.DataFrame(ctn.chay_e4(ctn.TacVu("E4", "viet_nam", 3, "fcm_kpp")))
    assert len(df) == 3 and df["kich_thuoc"].sum() == len(bo.y)
    assert set(bo.ten_cot) <= set(df.columns) and {"ti_le_nhan1", "mo_ho_%"} <= set(df.columns)


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
