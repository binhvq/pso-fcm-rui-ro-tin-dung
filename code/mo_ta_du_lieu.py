# -*- coding: utf-8 -*-
"""Hồ sơ chất lượng các bộ dữ liệu tín dụng (dùng cho mục 3.1 và bảng chẩn đoán 3.5).

Hai lớp đo:
- Dữ liệu GỐC (trước tiền xử lý): số dòng, số thuộc tính, loại thuộc tính (số liên tục / rời rạc ít giá trị), ô thiếu
  (kể cả số 0 giả ở bộ Việt Nam), dòng trùng, tỉ lệ nhãn 1 và tỉ số mất cân bằng.
- Dữ liệu SAU tiền xử lý (đúng thứ thuật toán nhìn thấy, tien_xu_ly.nap): số chiều, trung vị |độ lệch| và độ nhọn dư,
  tỉ lệ điểm có ít nhất một thuộc tính |z| > 3, số cặp thuộc tính |r| > 0,9 (thừa), số thành phần chính giữ 90% phương
  sai (số chiều hiệu dụng), trung vị số giá trị riêng mỗi thuộc tính.

Chạy: python3 mo_ta_du_lieu.py [file CSV kết quả]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

import tien_xu_ly as tx

NGUONG_RIEN_RAC = 12  # thuộc tính có ≤ 12 giá trị riêng coi là rời rạc (mã định danh / thứ bậc)


def _goc(ten: str):
    """Trả về (DataFrame thuộc tính gốc, nhãn gốc, số ô thiếu thực, số ô thiếu giả)."""
    raw = tx.RAW
    if ten == "viet_nam":
        v = pd.read_csv(glob.glob(os.path.join(raw, "vn-financial-distress-listed-firms", "*.csv"))[0], encoding="latin-1")
        f = v[[f"X{i}" for i in range(1, 26)]]
        return f, v["Financial_Distress"].values, int(f.isna().sum().sum()), int((f[["X6", "X7", "X12"]] == 0).sum().sum())
    if ten == "taiwan_default":
        t = pd.read_excel(glob.glob(os.path.join(raw, "taiwan-default", "*.xls"))[0], header=1)
        f = t.drop(columns=[t.columns[0], t.columns[-1]])
        return f, t.iloc[:, -1].values, int(f.isna().sum().sum()), 0
    if ten == "south_german":
        s = pd.read_csv(os.path.join(raw, "south-german-credit", "SouthGermanCredit.asc"), sep=r"\s+")
        y = 1 - s.pop("kredit").values
        return s, y, int(s.isna().sum().sum()), 0
    if ten == "german_so":
        a = np.loadtxt(os.path.join(raw, "german-credit", "german.data-numeric"))
        return pd.DataFrame(a[:, :-1]), (a[:, -1] == 2).astype(int), 0, 0
    if ten == "australian":
        a = np.loadtxt(os.path.join(raw, "australian-credit", "australian.dat"))
        return pd.DataFrame(a[:, :-1]), (a[:, -1] == 0).astype(int), 0, 0
    if ten == "polish":
        from scipy.io import arff
        d, _ = arff.loadarff(os.path.join(raw, "polish-bankruptcy", "3year.arff"))
        p = pd.DataFrame(d)
        y = p.pop("class").map(lambda b: int(b.decode() if isinstance(b, bytes) else b)).values
        p = p.apply(pd.to_numeric, errors="coerce")
        return p, y, int(p.isna().sum().sum()), 0
    if ten == "taiwan_pha_san":
        t = pd.read_csv(os.path.join(raw, "taiwan-bankruptcy", "data.csv"))
        y = t.pop("Bankrupt?").values
        return t, y, int(t.isna().sum().sum()), 0
    raise KeyError(ten)


BO = [  # (tên hồ sơ, tên bộ đã tiền xử lý, loại tín dụng)
    ("viet_nam", "viet_nam", "Doanh nghiệp niêm yết VN — khó khăn tài chính"),
    ("taiwan_default", "taiwan_kiem_chung", "Thẻ tín dụng cá nhân — vỡ nợ"),
    ("taiwan_default", "taiwan_kiem_chung_khong_log", "Thẻ tín dụng cá nhân — vỡ nợ (không log)"),
    ("south_german", "south_german", "Vay tiêu dùng cá nhân — tín dụng xấu"),
    ("german_so", "german_so", "Vay tiêu dùng cá nhân (bản số hoá gốc)"),
    ("australian", "australian", "Duyệt thẻ tín dụng — bị từ chối"),
    ("polish", "polish", "Doanh nghiệp Ba Lan — phá sản"),
    ("taiwan_pha_san", "taiwan_pha_san", "Doanh nghiệp Đài Loan — phá sản"),
]


def ho_so(ten_goc: str, ten_bo: str, loai: str) -> dict:
    f, y, thieu, thieu_gia = _goc(ten_goc)
    so_gia_tri = f.nunique()
    roi_rac = int((so_gia_tri <= NGUONG_RIEN_RAC).sum())
    ti_le = float(np.mean(y))
    bo = tx.nap(ten_bo)
    X = bo.X
    r = np.corrcoef(X, rowvar=False)
    cap_thua = int((np.abs(r[np.triu_indices_from(r, k=1)]) > 0.9).sum())
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    d90 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1)
    return {
        "bo": ten_bo, "loai": loai,
        "n_goc": len(f), "D_goc": f.shape[1], "so_lien_tuc": f.shape[1] - roi_rac, "so_roi_rac": roi_rac,
        "o_thieu_%": round(100 * thieu / f.size, 2), "o_0_gia_%": round(100 * thieu_gia / f.size, 2),
        "dong_trung": int(f.duplicated().sum()),
        "nhan1_%": round(100 * ti_le, 2), "mat_can_bang": round((1 - ti_le) / ti_le, 1),
        "n_dung": X.shape[0], "D_dung": X.shape[1],
        "lech_tv": round(float(np.median(np.abs(skew(X)))), 2), "nhon_tv": round(float(np.median(kurtosis(X))), 1),
        "ngoai_lai_%": round(100 * float(np.mean((np.abs(X) > 3).any(axis=1))), 1),
        "cap_r>0,9": cap_thua, "D_hieu_dung_90%": d90,
        "gia_tri_rieng_tv": int(np.median([len(np.unique(c)) for c in X.T])),
    }


if __name__ == "__main__":
    bang = pd.DataFrame([ho_so(*b) for b in BO])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(bang.to_string(index=False))
    if len(sys.argv) > 1:
        bang.to_csv(sys.argv[1], index=False)
