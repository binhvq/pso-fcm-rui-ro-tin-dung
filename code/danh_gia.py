# -*- coding: utf-8 -*-
"""Đánh giá một lượt chạy: mọi chỉ số của thực nghiệm chính thức từ (X, y, KetQua).

Hàm chỉ nhận kết quả đã chạy; nhãn y không bao giờ đi vào thuật toán, chỉ dùng để chấm (AUC tách rủi ro, chênh tỉ lệ,
ARI, NMI). Khoá và thứ tự khoá cố định theo COT (thứ tự cột CSV).

- J_m: J_m(U*(V), V) với U tối ưu theo V, m = 2, trên không gian của X — chỉ so trong loại I.
- PC, PE, XB, DB: trên (U, V) của chính thuật toán (chi_so_danh_gia).
- silhouette: nhãn cứng trên mẫu cố định (mau_silhouette, rng 0) để mọi phương pháp cùng mẫu; NaN nếu < 2 cụm.
- auc_rui_ro: cụm k* có tỉ lệ nhãn 1 cao nhất trong phân hoạch cứng; điểm rủi ro = U[:, k*]. Chọn k* bằng nhãn (như
  nhau cho mọi phương pháp) nhưng không tối đa hoá AUC: khi tâm gần trùng, U[:, k*] không tăng theo rủi ro nên AUC
  có thể dưới 0,5.
- auc_cung (chỉ số phụ, thêm sau lần chạy chính thức, chỉ mô tả): điểm rủi ro của điểm i là tỉ lệ nhãn 1 của cụm
  cứng chứa i — AUC của chính phân hoạch, luôn ≥ 0,5, không phụ thuộc độ thuộc.
- chenh_ti_le: tỉ lệ nhãn 1 cao nhất − thấp nhất giữa các cụm không rỗng.
- cum_nho_nhat: kích thước cụm cứng nhỏ nhất / n; so_cum_rong: số cụm không có điểm.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, silhouette_score

import chi_so_danh_gia as cs
from fcm import FCM
from giao_dien import KetQua

COT = ["J_m", "PC", "PE", "XB", "DB", "silhouette", "auc_rui_ro", "auc_cung", "chenh_ti_le", "cum_nho_nhat",
       "so_cum_rong", "ARI", "NMI", "n_fe", "n_vong_fcm", "t_khoi_tao", "t_tim_kiem", "t_tinh_chinh", "t_tong", "goi_pso"]
N_MAU_SILHOUETTE = 3000


def mau_silhouette(n: int, seed: int = 0) -> np.ndarray:
    """Chỉ số mẫu cố định (đã sắp xếp) để tính Silhouette; lấy toàn bộ khi n ≤ N_MAU_SILHOUETTE."""
    if n <= N_MAU_SILHOUETTE:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, N_MAU_SILHOUETTE, replace=False))


def danh_gia(X: np.ndarray, y: np.ndarray, kq: KetQua, m: float = 2.0, mau: np.ndarray | None = None) -> dict:
    mau = mau_silhouette(len(X)) if mau is None else mau
    c = kq.U.shape[1]
    nhan = kq.nhan
    kich_thuoc = np.bincount(nhan, minlength=c)
    khong_rong = np.flatnonzero(kich_thuoc)
    ti_le = np.array([y[nhan == k].mean() for k in khong_rong])
    k_rui_ro = int(khong_rong[np.argmax(ti_le)])
    ti_le_cum = np.zeros(c)
    ti_le_cum[khong_rong] = ti_le
    nhan_mau = nhan[mau]
    sil = silhouette_score(X[mau], nhan_mau) if len(np.unique(nhan_mau)) > 1 else float("nan")
    d = {
        "J_m": FCM(c, m=m).objective_from_centroids(X, kq.V),
        "PC": cs.pc(kq.U),
        "PE": cs.pe(kq.U),
        "XB": cs.xb(X, kq.V, kq.U, m),
        "DB": cs.db(X, kq.V, kq.U, m),
        "silhouette": float(sil),
        "auc_rui_ro": float(roc_auc_score(y, kq.U[:, k_rui_ro])),
        "auc_cung": float(roc_auc_score(y, ti_le_cum[nhan])),
        "chenh_ti_le": float(ti_le.max() - ti_le.min()),
        "cum_nho_nhat": float(kich_thuoc.min() / len(X)),
        "so_cum_rong": int(c - len(khong_rong)),
        "ARI": cs.ari(y, nhan),
        "NMI": cs.nmi(y, nhan),
        "n_fe": int(kq.n_fe),
        "n_vong_fcm": int(kq.n_vong_fcm),
        "t_khoi_tao": kq.t_khoi_tao,
        "t_tim_kiem": kq.t_tim_kiem,
        "t_tinh_chinh": kq.t_tinh_chinh,
        "t_tong": kq.t_tong,
        "goi_pso": kq.thong_tin.get("goi_pso", ""),
    }
    assert list(d) == COT
    return d
