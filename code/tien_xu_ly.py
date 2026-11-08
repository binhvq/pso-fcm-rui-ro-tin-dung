# -*- coding: utf-8 -*-
"""Nạp và tiền xử lý các bộ dữ liệu tín dụng dùng trong thực nghiệm (mô tả nguồn, giấy phép: data/README.md).

    bo = nap("viet_nam")          # BoDuLieu(X, y, ten_cot, ten, X_goc)
    bo.X                           # (n, D) đã chuẩn hoá z-score; nhãn y chỉ dùng để đánh giá
    bo.X_goc                       # cùng hàng, cùng cột với X, giá trị sau làm sạch nhưng trước log và z-score

Các bộ:
- viet_nam: khó khăn tài chính doanh nghiệp niêm yết (Tran và cộng sự 2022). Số 0 ở X6, X7, X12 là giá trị thiếu giả
  → điền trung vị phần khác 0; bỏ cột gần trùng X15 (r = 0,999 với X14), X18 (0,96 với X16), X21 (0,95 với X19),
  X3 (0,944 với X2); log có dấu; z-score.
- taiwan_kiem_chung / taiwan_thiet_ke: Taiwan Default (UCI 350), 2 mẫu phân tầng 5.000 không giao nhau. Mẫu thiết
  kế (rng 0) đã dùng trong thí nghiệm thử; mẫu kiểm chứng (rng 1, lấy từ phần còn lại) dùng cho kết quả chính.
  Hậu tố _khong_log: bỏ bước log (dùng ở mục 3.5).
- Bảng chẩn đoán số cực tiểu: south_german, german_so, australian, polish (năm thứ 3, mẫu 5.000), taiwan_pha_san
  (mẫu 5.000). Không log: south_german, german_so (mã thứ bậc), australian (đa số thuộc tính là mã định danh).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
N_MAU = 5000


@dataclass
class BoDuLieu:
    ten: str
    X: np.ndarray
    y: np.ndarray  # 1 = vỡ nợ / khó khăn tài chính / phá sản; chỉ dùng để đánh giá
    ten_cot: list[str]
    X_goc: np.ndarray  # giá trị gốc sau làm sạch, trước log và z-score; để đọc profile cụm


def signed_log(X: np.ndarray) -> np.ndarray:
    """sign(x)·log(1 + |x|): nén đuôi dài, giữ dấu, giữ 0."""
    return np.sign(X) * np.log1p(np.abs(X))


def zscore(X: np.ndarray, ten_cot: list[str]) -> tuple[np.ndarray, list[str]]:
    """Chuẩn hoá z-score theo cột; bỏ cột hằng (độ lệch chuẩn bằng 0)."""
    sd = X.std(axis=0)
    giu = sd > 0
    return (X[:, giu] - X[:, giu].mean(axis=0)) / sd[giu], [c for c, g in zip(ten_cot, giu) if g]


def mau_phan_tang(y: np.ndarray, n: int, rng: np.random.Generator, trong: np.ndarray | None = None) -> np.ndarray:
    """Chỉ số mẫu con phân tầng theo nhãn, giữ tỉ lệ lớp của toàn bộ y; chọn trong tập chỉ số `trong` nếu có."""
    trong = np.arange(len(y)) if trong is None else trong
    return np.concatenate([rng.choice(trong[y[trong] == k], int(round(n * np.mean(y == k))), replace=False)
                           for k in np.unique(y)])


def _hoan_tat(ten, X, y, ten_cot, log=True) -> BoDuLieu:
    X_goc = np.asarray(X, dtype=float)
    X = signed_log(X_goc) if log else X_goc
    giu = X.std(axis=0) > 0
    X, ten_cot = zscore(X, list(ten_cot))
    return BoDuLieu(ten, X, np.asarray(y).astype(int), ten_cot, X_goc[:, giu].copy())


def _viet_nam_bang(raw):
    v = pd.read_csv(glob.glob(os.path.join(raw, "vn-financial-distress-listed-firms", "*.csv"))[0], encoding="latin-1")
    f = v[[f"X{i}" for i in range(1, 26)]].copy()
    for col in ("X6", "X7", "X12"):
        f[col] = f[col].replace(0, np.nan)
        f[col] = f[col].fillna(f[col].median())
    f = f.drop(columns=["X15", "X18", "X21", "X3"])
    return f, v["Financial_Distress"].values, v["Company"].values


def _viet_nam(raw):
    f, y, _ = _viet_nam_bang(raw)
    return _hoan_tat("viet_nam", f.values, y, f.columns)


def viet_nam_nua(nua: int, raw: str = RAW) -> BoDuLieu:
    """Một nửa bộ Việt Nam theo doanh nghiệp (mẫu kiểm chứng độc lập cho E10): mã chứng khoán xáo bằng rng 0, nửa 0 là
    các mã ở nửa đầu. Mọi năm của một doanh nghiệp nằm cùng một nửa; log và z-score tính riêng trên nửa đó. Bước điền trung
    vị X6, X7, X12 vẫn dùng trung vị toàn bộ (không dùng nhãn)."""
    if nua not in (0, 1):
        raise ValueError("nua phải là 0 hoặc 1")
    f, y, cty = _viet_nam_bang(raw)
    ma = np.random.default_rng(0).permutation(np.unique(cty))
    chon = np.isin(cty, ma[: len(ma) // 2] if nua == 0 else ma[len(ma) // 2:])
    return _hoan_tat(f"viet_nam_nua_{nua}", f.values[chon], y[chon], f.columns)


def _taiwan(raw, mau, log):
    t = pd.read_excel(glob.glob(os.path.join(raw, "taiwan-default", "*.xls"))[0], header=1)
    y = t.iloc[:, -1].values
    f = t.drop(columns=[t.columns[0], t.columns[-1]])
    thiet_ke = mau_phan_tang(y, N_MAU, np.random.default_rng(0))
    if mau == "thiet_ke":
        idx = thiet_ke
    else:
        con_lai = np.setdiff1d(np.arange(len(y)), thiet_ke)
        idx = mau_phan_tang(y, N_MAU, np.random.default_rng(1), trong=con_lai)
    ten = f"taiwan_{mau}" + ("" if log else "_khong_log")
    return _hoan_tat(ten, f.values[idx], y[idx], f.columns, log=log)


def _south_german(raw):
    s = pd.read_csv(os.path.join(raw, "south-german-credit", "SouthGermanCredit.asc"), sep=r"\s+")
    y = 1 - s.pop("kredit").values  # kredit: 1 = tốt → nhãn rủi ro 1 = xấu
    return _hoan_tat("south_german", s.values, y, s.columns, log=False)


def _german_so(raw):
    a = np.loadtxt(os.path.join(raw, "german-credit", "german.data-numeric"))
    return _hoan_tat("german_so", a[:, :-1], (a[:, -1] == 2), [f"A{i}" for i in range(1, a.shape[1])], log=False)


def _australian(raw):
    a = np.loadtxt(os.path.join(raw, "australian-credit", "australian.dat"))
    # Nhãn A15: 1 = được duyệt; coi 0 (bị từ chối) là nhóm rủi ro
    return _hoan_tat("australian", a[:, :-1], (a[:, -1] == 0), [f"A{i}" for i in range(1, a.shape[1])], log=False)


def _lam_sach_so(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.nunique(dropna=True) > 1]
    return df.fillna(df.median())


def _polish(raw):
    from scipy.io import arff

    data, _ = arff.loadarff(os.path.join(raw, "polish-bankruptcy", "3year.arff"))
    p = pd.DataFrame(data)
    y = p.pop("class").map(lambda b: int(b.decode() if isinstance(b, bytes) else b)).values
    f = _lam_sach_so(p)
    idx = mau_phan_tang(y, N_MAU, np.random.default_rng(0))
    return _hoan_tat("polish", f.values[idx], y[idx], f.columns)


def _taiwan_pha_san(raw):
    t = pd.read_csv(os.path.join(raw, "taiwan-bankruptcy", "data.csv"))
    y = t.pop("Bankrupt?").values
    f = _lam_sach_so(t)
    idx = mau_phan_tang(y, N_MAU, np.random.default_rng(0))
    return _hoan_tat("taiwan_pha_san", f.values[idx], y[idx], [c.strip() for c in f.columns])


_BO = {
    "viet_nam": _viet_nam,
    "taiwan_kiem_chung": lambda raw: _taiwan(raw, "kiem_chung", log=True),
    "taiwan_kiem_chung_khong_log": lambda raw: _taiwan(raw, "kiem_chung", log=False),
    "taiwan_thiet_ke": lambda raw: _taiwan(raw, "thiet_ke", log=True),
    "south_german": _south_german,
    "german_so": _german_so,
    "australian": _australian,
    "polish": _polish,
    "taiwan_pha_san": _taiwan_pha_san,
}
DANH_SACH = tuple(_BO)


def nap(ten: str, raw: str = RAW) -> BoDuLieu:
    if ten not in _BO:
        raise KeyError(f"không có bộ '{ten}'; chọn trong {DANH_SACH}")
    return _BO[ten](raw)
