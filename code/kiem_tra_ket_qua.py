# -*- coding: utf-8 -*-
"""Kiểm tra tự động trên kết quả thực nghiệm (K1–K8 chốt trước khi chạy; K9 thêm cùng chỉ số phụ auc_cung).

    python3 kiem_tra_ket_qua.py [--ket-qua DIR] [--nhanh] > results/kiem-tra.txt

K1–K3, K7, K9 bắt buộc (thất bại = lỗi code). K4–K6, K8 là kiểm tra hợp lý: thất bại nghĩa là kết quả khác bản thử —
không sửa code để khớp, ghi lại và báo người dùng. Mã thoát 1 khi có điều kiện bắt buộc không đạt, 2 khi chỉ có điều
kiện hợp lý không đạt.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import chay_thuc_nghiem as ctn
from danh_gia import COT

TOL_J = 1e-4
LOAI_I_CUNG_FCM = ["b_qpso_fcm", "cach2_fcm_gieo", "cach3_xen_ke", "fcm_lap_lai", "fcm_kpp", "de_xuat"]
BO_SUP = ["south_german", "german_so", "taiwan_pha_san"]
TUY_CHON = {"silhouette", "goi_pso"}


def doc(thu_muc: str, thi_nghiem: str) -> pd.DataFrame:
    tep = sorted(glob.glob(os.path.join(thu_muc, thi_nghiem, "*.csv")))
    return pd.concat([pd.read_csv(t) for t in tep], ignore_index=True) if tep else pd.DataFrame()


def la_true(s: pd.Series) -> pd.Series:
    return s.astype(str) == "True"


def ti_le_dat_j_tot_nhat(df: pd.DataFrame) -> pd.Series:
    """Tỉ lệ lượt mỗi phương pháp có J ≤ (1 + TOL_J)·J nhỏ nhất của mọi lượt mọi phương pháp cùng (bộ, c)."""
    j_min = df.groupby(["bo", "c"])["J_m"].transform("min")
    return (df["J_m"] <= (1 + TOL_J) * j_min).groupby([df["bo"], df["c"], df["phuong_phap"]]).mean()


def kiem_tra(thu_muc: str, nhanh: bool = False) -> list[tuple[str, bool, bool, str]]:
    """Trả danh sách (mã, bắt buộc, đạt, chi tiết)."""
    so_seed = ctn.SO_SEED_NHANH if nhanh else ctn.SO_SEED
    so_kt = ctn.SO_KHOI_TAO_NHANH if nhanh else ctn.SO_KHOI_TAO
    e1, e2, e3 = doc(thu_muc, "E1"), doc(thu_muc, "E2"), doc(thu_muc, "E3")
    kq = []

    # K1
    can = dict(E1=len(ctn.CAU_HINH["E1"]) * len(ctn.PHUONG_PHAP["E1"]) * so_seed,
               E2=len(ctn.CAU_HINH["E2"]) * len(ctn.PHUONG_PHAP["E2"]) * so_seed,
               E3=len(ctn.CAU_HINH["E3"]) * 2 * so_kt)
    co = dict(E1=len(e1), E2=len(e2), E3=len(e3))
    e3_moi = e3.groupby(["bo", "c"]).size() if len(e3) else pd.Series(dtype=int)
    n_e4 = len(glob.glob(os.path.join(thu_muc, "E4", "*.csv")))
    trung = sum(int(d.duplicated(["bo", "c", "seed", "phuong_phap"]).sum()) for d in (e1, e2) if len(d))
    dat = co == can and (e3_moi == 2 * so_kt).all() and n_e4 == 6 and trung == 0
    kq.append(("K1", True, dat, f"số dòng {co}, cần {can}; E3 mỗi (bộ, c) {sorted(set(e3_moi))} (cần {2 * so_kt}); "
                                f"E4 {n_e4}/6 tệp; khoá trùng {trung}"))

    # K2
    chi_tiet, dat = [], True
    for ten, d in (("E1", e1), ("E2", e2)):
        loi = d["loi"].notna() & (d["loi"].astype(str) != "")
        nan = d[[c for c in COT if c not in TUY_CHON]].isna().any(axis=1)
        dat &= not loi.any() and not nan.any()
        chi_tiet.append(f"{ten}: {int(loi.sum())} dòng lỗi, {int(nan.sum())} dòng NaN ở cột bắt buộc, "
                        f"silhouette NaN {int(d['silhouette'].isna().sum())}")
    nan3 = int(e3[["J_m", "PC", "sup", "n_vong"]].isna().any(axis=1).sum())
    dat &= nan3 == 0
    kq.append(("K2", True, dat, "; ".join(chi_tiet) + f"; E3 NaN {nan3}"))

    # K3
    tv = ctn.TacVu("E1", "taiwan_kiem_chung", 3, ctn.SEED_DAU)
    moi = pd.DataFrame(ctn.chay_tac_vu(tv)[0]).set_index("phuong_phap")
    cu = e1[(e1["bo"] == tv.bo) & (e1["c"] == tv.c) & (e1["seed"] == tv.khoa)].set_index("phuong_phap").loc[moi.index]
    so = [c for c in COT if not c.startswith("t_") and c != "goi_pso"]
    lech = {c: float(np.nanmax(np.abs(moi[c].astype(float) - cu[c].astype(float)) /
                               np.fmax(np.abs(cu[c].astype(float)), 1e-300))) for c in so}
    nan_khop = all((moi[c].isna() == cu[c].isna()).all() for c in so)
    goi_khop = (moi["goi_pso"].astype(str).replace("", "nan") == cu["goi_pso"].astype(str)).all()
    dat = max(lech.values()) <= 1e-12 and nan_khop and goi_khop
    kq.append(("K3", True, dat, f"chạy lại {tv}: lệch tương đối lớn nhất {max(lech.values()):.2e} "
                                f"(cột {max(lech, key=lech.get)}), NaN khớp {nan_khop}, goi_pso khớp {goi_khop}"))

    # K4
    goc = e1[e1["phuong_phap"] == "fcm_kpp"].set_index(["bo", "c", "seed"])["J_m"]
    chi_tiet, dat = [], True
    for pp in LOAI_I_CUNG_FCM:
        d = e1[e1["phuong_phap"] == pp].set_index(["bo", "c", "seed"])["J_m"]
        cung = (np.abs(d - goc.loc[d.index]) <= TOL_J * goc.loc[d.index]).groupby(level=["bo", "c"]).mean()
        dat &= bool((cung >= 0.9).all())
        chi_tiet.append(f"{pp} min {cung.min():.0%}" + ("" if (cung >= 0.9).all() else
                                                         f" (không đạt: {cung[cung < 0.9].round(2).to_dict()})"))
    kq.append(("K4", False, dat, "tỉ lệ lượt cùng J_m với fcm_kpp mỗi (bộ, c): " + "; ".join(chi_tiet)))

    # K5
    dx = e1[e1["phuong_phap"] == "de_xuat"]
    n_goi = int(la_true(dx["goi_pso"]).sum())
    kq.append(("K5", False, n_goi == 0, f"de_xuat gọi PSO ở {n_goi}/{len(dx)} lượt E1"))

    # K6
    tl = ti_le_dat_j_tot_nhat(e2)
    tl6 = tl.xs(6, level="c").droplevel("bo")
    dat = 0.15 <= tl6["fcm"] <= 0.50 and tl6["fcm_kpp"] >= 0.90
    kq.append(("K6", False, dat, "E2 c = 6 tỉ lệ đạt J tốt nhất: " + ", ".join(f"{p} {v:.0%}" for p, v in tl6.items())
               + "; c = 3: " + ", ".join(f"{p} {v:.0%}" for p, v in tl.xs(3, level="c").droplevel("bo").items())))

    # K7
    ep = e2[e2["phuong_phap"] == "de_xuat_ep_pso"]
    dat = bool(la_true(ep["goi_pso"]).all() and (ep["n_fe"] > 0).all() and len(ep) > 0)
    kq.append(("K7", True, dat, f"de_xuat_ep_pso: goi_pso True {int(la_true(ep['goi_pso']).sum())}/{len(ep)}, "
                                f"n_fe > 0 {int((ep['n_fe'] > 0).sum())}/{len(ep)}"))

    # K8
    d = e3[(e3["kieu"] == "u_ngau_nhien") & e3["bo"].isin(BO_SUP) & e3["c"].between(2, 4)]
    sup = d.groupby(["bo", "c"])["sup"].mean()
    kq.append(("K8", False, bool((sup >= 0.9).all() and len(sup) == 3 * len(BO_SUP)),
               "tỉ lệ sụp khi khởi tạo U ngẫu nhiên: " + ", ".join(f"{b} c={c} {v:.0%}" for (b, c), v in sup.items())))
    # K9
    dat, chi_tiet = True, []
    for ten, d in (("E1", e1), ("E2", e2)):
        if "auc_cung" not in d:
            dat = False
            chi_tiet.append(f"{ten}: thiếu cột auc_cung")
            continue
        a = d["auc_cung"]
        dat &= bool(a.notna().all() and (a >= 0.5 - 1e-12).all())
        chi_tiet.append(f"{ten}: NaN {int(a.isna().sum())}, < 0,5 {int((a < 0.5 - 1e-12).sum())}, nhỏ nhất {a.min():.3f}")
    kq.append(("K9", True, dat, "AUC cứng (chỉ số phụ) ≥ 0,5 theo xây dựng: " + "; ".join(chi_tiet)))
    return kq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ket-qua", default=ctn.KET_QUA_MAC_DINH)
    ap.add_argument("--nhanh", action="store_true")
    a = ap.parse_args(argv)
    kq = kiem_tra(os.path.abspath(a.ket_qua), a.nhanh)
    for ma, bat_buoc, dat, chi_tiet in kq:
        print(f"{ma} [{'bắt buộc' if bat_buoc else 'hợp lý'}] {'ĐẠT' if dat else 'KHÔNG ĐẠT'}: {chi_tiet}")
    loi_bb = [ma for ma, bb, dat, _ in kq if bb and not dat]
    loi_hl = [ma for ma, bb, dat, _ in kq if not bb and not dat]
    print(f"\nBắt buộc không đạt: {loi_bb or 'không'}; hợp lý không đạt: {loi_hl or 'không'}")
    return 1 if loi_bb else (2 if loi_hl else 0)


if __name__ == "__main__":
    raise SystemExit(main())
