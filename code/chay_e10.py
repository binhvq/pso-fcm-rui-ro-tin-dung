# -*- coding: utf-8 -*-
"""Thí nghiệm E10: trọng số thuộc tính học trên một mẫu có giữ lợi ích khi áp lên mẫu độc lập không.

    python3 chay_e10.py [--tien-trinh 12] [--ket-qua DIR] [--chi-phan-tich]

Bốn chiều chuyển: Taiwan Default mẫu thiết kế → mẫu kiểm chứng và ngược lại; Việt Nam nửa 0 → nửa 1 và ngược lại (chia
theo doanh nghiệp, tien_xu_ly.viet_nam_nua). Mỗi (chiều, c ∈ {2, 3, 4}, seed): học trọng số trên mẫu nguồn A bằng PSO-W,
tìm ngẫu nhiên (cùng 500 lần tính fitness) và WFCM giải tích; trên mẫu đích B chạy FCM từ cùng một bộ tâm k-means++ V0 với
5 cách: trọng số bằng nhau (FCM thường), ba trọng số chuyển từ A, và PSO-W học ngay trên B (tham chiếu "tại chỗ", cùng
V0). Mọi phương pháp nhận default_rng(seed) riêng nên V0 trên B như nhau. Nhãn không dùng ở bước học. Ghi
<ket-qua>/E10/e10.csv; chạy lại chỉ chạy phần thiếu. Phân tích ghi <ket-qua>/bang/e10-*.tex và <ket-qua>/so-lieu-e10.tex.
Chỉ số chính như E8: auc_cung, chenh_ti_le trên B; thêm fitness của từng trọng số trên B (−Silhouette + phạt).
"""

from __future__ import annotations

import os

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tien_xu_ly as tx  # noqa: E402
from cach_ket_hop import bai_toan_trong_so, phan_hoach_trong_so, wfcm_giai_tich  # noqa: E402
from chay_thuc_nghiem import KET_QUA_MAC_DINH, SEED_DAU, SO_SEED  # noqa: E402
from danh_gia import danh_gia, mau_silhouette  # noqa: E402
from giao_dien import KetQua  # noqa: E402
from phan_tich import DAU_KL, _bang_float, ghi_so_lieu, so_sanh_cap, so_viet, ten_macro, tex  # noqa: E402
from pso_trong_so import ngau_nhien_trong_so, pso_trong_so  # noqa: E402

CHIEU = {"taiwan_tk_kc": ("taiwan_thiet_ke", "taiwan_kiem_chung"), "taiwan_kc_tk": ("taiwan_kiem_chung", "taiwan_thiet_ke"),
         "vn_0_1": ("viet_nam_nua_0", "viet_nam_nua_1"), "vn_1_0": ("viet_nam_nua_1", "viet_nam_nua_0")}
SO_CUM = (2, 3, 4)
M = 2.0
HOC = {"wfcm_chuyen": wfcm_giai_tich, "ngau_nhien_chuyen": ngau_nhien_trong_so, "pso_chuyen": pso_trong_so}
PHUONG_PHAP = ["fcm_kpp", "wfcm_chuyen", "ngau_nhien_chuyen", "pso_chuyen", "pso_tai_cho"]
KHOA = ["chieu", "c", "seed"]


@lru_cache(maxsize=None)
def nap(ten: str) -> tx.BoDuLieu:
    return tx.viet_nam_nua(int(ten[-1])) if ten.startswith("viet_nam_nua_") else tx.nap(ten)


def lam(tac_vu):
    chieu, c, seed = tac_vu
    A, B = (nap(t) for t in CHIEU[chieu])
    assert A.ten_cot == B.ten_cot, chieu
    mau = mau_silhouette(len(B.X))
    V0, fitness_B = bai_toan_trong_so(B.X, c, M, np.random.default_rng(seed))
    trong_so = {"fcm_kpp": np.ones(B.X.shape[1])}
    trong_so.update({ten: ham(A.X, c, M, np.random.default_rng(seed), None).thong_tin["trong_so"] for ten, ham in HOC.items()})
    tai_cho = pso_trong_so(B.X, c, M, np.random.default_rng(seed), None)
    dong = []
    for ten in PHUONG_PHAP:
        if ten == "pso_tai_cho":
            kq, w = tai_cho, tai_cho.thong_tin["trong_so"]
        else:
            w = trong_so[ten]
            U, V, n_vong, t = phan_hoach_trong_so(B.X, M, V0, w)
            kq = KetQua(U=U, V=V, n_vong_fcm=n_vong, t_tinh_chinh=t)
        dong.append(dict(chieu=chieu, nguon=CHIEU[chieu][0], dich=CHIEU[chieu][1], c=c, seed=seed, phuong_phap=ten,
                         fitness_dich=fitness_B(w), trong_so=";".join(f"{v:.6g}" for v in w),
                         **danh_gia(B.X, B.y, kq, M, mau)))
    return dong


def chay(thu_muc: str, so_tien_trinh: int) -> int:
    tep = os.path.join(thu_muc, "E10", "e10.csv")
    os.makedirs(os.path.dirname(tep), exist_ok=True)
    cu = pd.read_csv(tep) if os.path.exists(tep) else None
    xong = set() if cu is None else set(zip(cu["chieu"], cu["c"], cu["seed"]))
    viec = [(h, c, SEED_DAU + s) for h in CHIEU for c in sorted(SO_CUM, reverse=True) for s in range(SO_SEED)
            if (h, c, SEED_DAU + s) not in xong]
    if viec:
        with Pool(so_tien_trinh) as p:
            for i, ds in enumerate(p.imap_unordered(lam, viec), 1):
                moi = pd.DataFrame(ds)
                cu = moi if cu is None else pd.concat([cu, moi], ignore_index=True)
                cu.sort_values(KHOA + ["phuong_phap"]).to_csv(tep, index=False)
                print(f"[{i}/{len(viec)}] {ds[0]['chieu']} c={ds[0]['c']} {ds[0]['seed']}", flush=True)
    return len(viec)


# ------------------------------------------------------------------ phân tích
TEN_CHIEU = {"taiwan_tk_kc": "Taiwan: thiết kế → kiểm chứng", "taiwan_kc_tk": "Taiwan: kiểm chứng → thiết kế",
             "vn_0_1": "Việt Nam: nửa 0 → nửa 1", "vn_1_0": "Việt Nam: nửa 1 → nửa 0"}
NGAN = {"fcm_kpp": "FCM", "wfcm_chuyen": "WFCM", "ngau_nhien_chuyen": "NN", "pso_chuyen": "PSO", "pso_tai_cho": "Tại chỗ"}
DOI_CHUNG = ["fcm_kpp", "wfcm_chuyen", "ngau_nhien_chuyen", "pso_tai_cho"]
CHI_SO = ["auc_cung", "chenh_ti_le", "fitness_dich"]


def doc(thu_muc: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(thu_muc, "E10", "e10.csv"))


def kiem_dinh(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["fitness_dich"] = -x["fitness_dich"]  # đổi dấu để "lớn hơn là tốt hơn" như 2 chỉ số kia
    return so_sanh_cap(x, ["chieu", "c"], "pso_chuyen", DOI_CHUNG, CHI_SO)


def bang_ket_qua(d: pd.DataFrame, kd: pd.DataFrame) -> str:
    tv = d.groupby(["chieu", "c", "phuong_phap"])[["auc_cung", "chenh_ti_le"]].median()
    than = []
    for h in CHIEU:
        for i, c in enumerate(SO_CUM):
            o = [tex(TEN_CHIEU[h]) if i == 0 else "", str(c)]
            for cs in ("auc_cung", "chenh_ti_le"):
                cao = max(tv.loc[(h, c, pp), cs] for pp in PHUONG_PHAP)
                for pp in PHUONG_PHAP:
                    s = so_viet(tv.loc[(h, c, pp), cs], 3)
                    if pp != "pso_chuyen":
                        k = kd[(kd["chieu"] == h) & (kd["c"] == c) & (kd["chi_so"] == cs) & (kd["doi_chung"] == pp)]
                        s += DAU_KL[int(k["ket_luan"].iloc[0])]
                    o.append(f"\\textbf{{{s}}}" if tv.loc[(h, c, pp), cs] >= cao - 5e-4 else s)
            than.append(" & ".join(o) + " \\\\")
        if h != list(CHIEU)[-1]:
            than.append("\\midrule")
    ten = " & ".join(NGAN[pp] for pp in PHUONG_PHAP)
    tieu_de = ["& & \\multicolumn{5}{c}{AUC phân hoạch trên mẫu đích} & \\multicolumn{5}{c}{Chênh tỉ lệ vỡ nợ trên mẫu đích} \\\\",
               "\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}", f"Chiều chuyển & c & {ten} & {ten} \\\\"]
    return _bang_float("@{}llrrrrrrrrrr@{}", tieu_de, than,
                       "TN5 -- trọng số học trên mẫu nguồn, áp lên mẫu đích độc lập", "tab:e10-ket-qua",
                       "Trung vị 30 seed, m = 2. FCM: trọng số bằng nhau; WFCM, NN, PSO: trọng số học trên mẫu nguồn bằng "
                       "WFCM giải tích, tìm ngẫu nhiên và PSO-W; Tại chỗ: PSO-W học ngay trên mẫu đích. Mọi cột dùng cùng "
                       "tâm khởi tạo k-means++ trên mẫu đích. Đậm: cao nhất hàng. Dấu \\textsuperscript{+} / "
                       "\\textsuperscript{--} sau một cột: PSO chuyển tốt hơn / kém hơn cột đó có ý nghĩa (Wilcoxon theo "
                       "cặp seed, Holm toàn họ, $\\alpha = 0{,}05$).", tabcolsep=2)


def bang_tong_hop(kd: pd.DataFrame) -> str:
    ten_cs = {"auc_cung": "AUC phân hoạch", "chenh_ti_le": "Chênh tỉ lệ vỡ nợ", "fitness_dich": "Silhouette − phạt"}
    than = []
    for cs in CHI_SO:
        o = [tex(ten_cs[cs])]
        for dc in DOI_CHUNG:
            k = kd[(kd["chi_so"] == cs) & (kd["doi_chung"] == dc)]["ket_luan"]
            o.append(f"{(k == 1).sum()} / {(k == 0).sum()} / {(k == -1).sum()}")
        than.append(" & ".join(o) + " \\\\")
    tieu_de = ["PSO chuyển so với & FCM & WFCM chuyển & NN chuyển & PSO tại chỗ \\\\"]
    return _bang_float("@{}lcccc@{}", tieu_de, than,
                       "TN5 -- số cấu hình trọng số PSO chuyển sang mẫu đích tốt hơn / không khác / kém hơn từng cách khác",
                       "tab:e10-tong-hop",
                       f"{kd.groupby(['chieu', 'c']).ngroups} cấu hình (4 chiều chuyển × 3 số cụm), Wilcoxon theo cặp "
                       "seed, Holm toàn họ, $\\alpha = 0{,}05$.")


def macro(d: pd.DataFrame, kd: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuEMuoiSoLuot", so_viet(len(d), 0)), ("soLieuEMuoiSoCauHinh", str(d.groupby(["chieu", "c"]).ngroups)),
          ("soLieuEMuoiSoPhep", str(len(kd)))]
    for cs in CHI_SO:
        for dc in DOI_CHUNG:
            k = kd[(kd["chi_so"] == cs) & (kd["doi_chung"] == dc)]["ket_luan"]
            for ten, v in (("Hon", 1), ("Bang", 0), ("Kem", -1)):
                ra.append(("soLieu" + ten_macro("EMuoi", cs, dc, ten), str(int((k == v).sum()))))
    tv = d.groupby(["chieu", "c", "phuong_phap"])[["auc_cung", "chenh_ti_le", "fitness_dich"]].median()
    for (h, c, pp), r in tv.iterrows():
        for cs in ("auc_cung", "chenh_ti_le"):
            ra.append(("soLieu" + ten_macro("EMuoi", h, "C", c, pp, cs), so_viet(r[cs], 3)))
    for h in CHIEU:  # độ giống nhau giữa trọng số PSO học trên nguồn và học tại chỗ trên đích (cosine, trung vị)
        x = d[d["chieu"] == h].pivot_table(index=["c", "seed"], columns="phuong_phap", values="trong_so", aggfunc="first")
        cos = [np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in
               ((np.array(r["pso_chuyen"].split(";"), float), np.array(r["pso_tai_cho"].split(";"), float))
                for _, r in x.iterrows())]
        ra.append(("soLieu" + ten_macro("EMuoi", h, "CosTrongSo"), so_viet(np.median(cos), 2)))
    return ra


def phan_tich(thu_muc: str) -> None:
    d = doc(thu_muc)
    kd = kiem_dinh(d)
    kd.to_csv(os.path.join(thu_muc, "E10", "kiem-dinh.csv"), index=False)
    os.makedirs(os.path.join(thu_muc, "bang"), exist_ok=True)
    for ten, noi_dung in (("e10-ket-qua", bang_ket_qua(d, kd)), ("e10-tong-hop", bang_tong_hop(kd))):
        with open(os.path.join(thu_muc, "bang", f"{ten}.tex"), "w", encoding="utf-8") as f:
            f.write(noi_dung)
    ghi_so_lieu(macro(d, kd), os.path.join(thu_muc, "so-lieu-e10.tex"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tien-trinh", type=int, default=12)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--chi-phan-tich", action="store_true")
    a = ap.parse_args(argv)
    if not a.chi_phan_tich:
        t0 = time.perf_counter()
        n = chay(os.path.abspath(a.ket_qua), a.tien_trinh)
        print(f"E10: {n} tác vụ, {time.perf_counter() - t0:.0f} s", flush=True)
    phan_tich(os.path.abspath(a.ket_qua))


if __name__ == "__main__":
    main()
