# -*- coding: utf-8 -*-
"""Mốc tham chiếu cho TN3: K-means cứng không trọng số trong đúng khung E8.

    python3 chay_e8_kmeans.py [--tien-trinh 8] [--ket-qua DIR] [--chi-phan-tich]

Cùng 7 bộ, c ∈ {2, 3, 4}, 30 seed, default_rng(seed) và mẫu Silhouette như E8. Ghi <ket-qua>/E8/kmeans.csv.
So PSO tìm trọng số (lấy từ e8.csv) với K-means trên auc_cung và chenh_ti_le: Wilcoxon theo cặp seed ở từng
(bộ, c, chỉ số), Holm trên họ riêng của phép so này để không đổi họ kiểm định đã báo của TN3.
Ghi <ket-qua>/E8/kiem-dinh-kmeans.csv, <ket-qua>/bang/e8-kmeans.tex và <ket-qua>/so-lieu-e8-kmeans.tex.
"""

from __future__ import annotations

import os

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chay_e8 import ALPHA, BO, KHOA, M, SO_CUM, _ten_bo, doc  # noqa: E402
from chay_thuc_nghiem import KET_QUA_MAC_DINH, SEED_DAU, SO_SEED, nap  # noqa: E402
from danh_gia import danh_gia, mau_silhouette  # noqa: E402
from giao_dien import kmeans  # noqa: E402
from phan_tich import _bang_float, ghi_so_lieu, holm, so_viet, ten_macro, tex, wilcoxon_cap  # noqa: E402

CHI_SO = {"auc_cung": "AUC phân hoạch", "chenh_ti_le": "Chênh tỉ lệ vỡ nợ"}
BO_CHINH = ["viet_nam", "taiwan_kiem_chung"]


def lam(tac_vu):
    bo, c, seed = tac_vu
    d = nap(bo)
    kq = kmeans(d.X, c, M, np.random.default_rng(seed), None)
    return dict(bo=bo, c=c, seed=seed, phuong_phap="kmeans", **danh_gia(d.X, d.y, kq, M, mau_silhouette(len(d.X))))


def chay(thu_muc: str, so_tien_trinh: int) -> int:
    tep = os.path.join(thu_muc, "E8", "kmeans.csv")
    cu = pd.read_csv(tep) if os.path.exists(tep) else None
    xong = set() if cu is None else set(zip(cu["bo"], cu["c"], cu["seed"]))
    viec = [(b, c, SEED_DAU + s) for b in BO for c in SO_CUM for s in range(SO_SEED) if (b, c, SEED_DAU + s) not in xong]
    if viec:
        with Pool(so_tien_trinh) as p:
            moi = pd.DataFrame(p.map(lam, viec))
        cu = moi if cu is None else pd.concat([cu, moi], ignore_index=True)
        cu.sort_values(KHOA).to_csv(tep, index=False)
    return len(viec)


def kiem_dinh(pso: pd.DataFrame, km: pd.DataFrame) -> pd.DataFrame:
    """Mỗi (bộ, c, chỉ số): trung vị hiệu PSO − K-means theo cặp seed, p Wilcoxon, p Holm trong họ này, kết luận
    +1 (PSO tốt hơn có ý nghĩa), −1 (kém hơn), 0 (không phát hiện khác biệt)."""
    hang = []
    for (b, c), g in pso.groupby(["bo", "c"]):
        k = km[(km["bo"] == b) & (km["c"] == c)].set_index("seed")
        g = g.set_index("seed")
        for cs in CHI_SO:
            hieu = (g[cs] - k[cs]).dropna()
            if len(hieu) != SO_SEED:
                raise ValueError(f"{b} c={c} {cs}: chỉ ghép được {len(hieu)}/{SO_SEED} seed")
            hang.append(dict(bo=b, c=c, chi_so=cs, hieu_tv=float(hieu.median()),
                             p=wilcoxon_cap(hieu.to_numpy(), np.zeros(len(hieu)))))
    kd = pd.DataFrame(hang)
    kd["p_holm"] = holm(kd["p"].to_numpy())
    kd["ket_luan"] = np.where(kd["p_holm"] < ALPHA, np.sign(kd["hieu_tv"]), 0).astype(int)
    return kd


def bang(tv: pd.DataFrame, kd: pd.DataFrame) -> str:
    than = []
    for b in BO_CHINH:
        for i, c in enumerate(SO_CUM):
            o = [tex(_ten_bo(b)) if i == 0 else "", str(c)]
            for cs in CHI_SO:
                k = int(kd[(kd["bo"] == b) & (kd["c"] == c) & (kd["chi_so"] == cs)]["ket_luan"].iloc[0])
                o += [so_viet(tv.loc[(b, c, "fcm_kpp"), cs], 3), so_viet(tv.loc[(b, c, "kmeans"), cs], 3),
                      so_viet(tv.loc[(b, c, "pso_trong_so"), cs], 3)
                      + {1: "\\textsuperscript{+}", -1: "\\textsuperscript{--}", 0: ""}[k]]
            than.append(" & ".join(o) + " \\\\")
        if b != BO_CHINH[-1]:
            than.append("\\midrule")
    tieu_de = ["& & \\multicolumn{3}{c}{AUC phân hoạch} & \\multicolumn{3}{c}{Chênh tỉ lệ vỡ nợ} \\\\",
               "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
               "Bộ dữ liệu & c & FCM & K-means & PSO & FCM & K-means & PSO \\\\"]
    return _bang_float("@{}llrrrrrr@{}", tieu_de, than,
                       "TN3 -- K-means không trọng số làm mốc tham chiếu trên 2 bộ chính", "tab:e8-kmeans",
                       "Trung vị 30 seed, cùng seed và tiền xử lý với Bảng~\\ref{tab:e8-ket-qua}. FCM: FCM k-means++, "
                       "m = 2; PSO: FCM với trọng số do PSO tìm. Dấu \\textsuperscript{+} / \\textsuperscript{--}: PSO "
                       "tốt hơn / kém hơn K-means có ý nghĩa (Wilcoxon theo cặp seed, Holm trên 42 phép so với K-means "
                       "ở 7 bộ, $\\alpha = 0{,}05$).")


def macro(tv: pd.DataFrame, kd: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuETamKmSoPhep", str(len(kd)))]
    for cs in CHI_SO:
        for pham_vi, loc in (("", kd), ("Chinh", kd[kd["bo"].isin(BO_CHINH)])):
            k = loc[loc["chi_so"] == cs]["ket_luan"]
            for ten, v in (("Hon", 1), ("Bang", 0), ("Kem", -1)):
                ra.append(("soLieu" + ten_macro("ETamKm", pham_vi, cs, ten), str(int((k == v).sum()))))
    for b in BO:
        for c in SO_CUM:
            for cs in CHI_SO:
                ra.append(("soLieu" + ten_macro("ETamKm", b, "C", c, cs), so_viet(tv.loc[(b, c, "kmeans"), cs], 3)))
    return ra


def phan_tich(thu_muc: str) -> None:
    pso = doc(thu_muc)
    km = pd.read_csv(os.path.join(thu_muc, "E8", "kmeans.csv"))
    kd = kiem_dinh(pso[pso["phuong_phap"] == "pso_trong_so"], km)
    kd.to_csv(os.path.join(thu_muc, "E8", "kiem-dinh-kmeans.csv"), index=False)
    tv = pd.concat([pso, km], ignore_index=True).groupby(["bo", "c", "phuong_phap"])[list(CHI_SO)].median()
    with open(os.path.join(thu_muc, "bang", "e8-kmeans.tex"), "w", encoding="utf-8") as f:
        f.write(bang(tv, kd))
    ghi_so_lieu(macro(tv, kd), os.path.join(thu_muc, "so-lieu-e8-kmeans.tex"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tien-trinh", type=int, default=8)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--chi-phan-tich", action="store_true")
    a = ap.parse_args(argv)
    if not a.chi_phan_tich:
        t0 = time.perf_counter()
        n = chay(os.path.abspath(a.ket_qua), a.tien_trinh)
        print(f"E8 K-means: {n} tác vụ, {time.perf_counter() - t0:.0f} s", flush=True)
    phan_tich(os.path.abspath(a.ket_qua))


if __name__ == "__main__":
    main()
