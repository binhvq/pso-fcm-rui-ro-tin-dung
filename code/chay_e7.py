# -*- coding: utf-8 -*-
"""Thí nghiệm E7: 5 phiên bản PSO làm bộ khởi tạo FCM trên các cấu hình có nhiều cực tiểu (m = 1,5) và lưu quỹ đạo gbest.

    python3 chay_e7.py [--tien-trinh 12] [--ket-qua DIR] [--hinh DIR] [--chi-phan-tich]

Cấu hình: Taiwan Default m = 1,5, c = 7 và Việt Nam m = 1,5, c = 8 (2 cấu hình nhiều cực tiểu của E6), cùng 2 cấu hình
chính của E5 (m = 2, c = 3) để có quỹ đạo gbest. Mỗi (cấu hình, seed) chạy tuần tự: 3 đối chứng của E5, FCM k-means++
chạy lại 20 lần lấy J nhỏ nhất (đối chứng cùng cỡ chi phí), và 5 phiên bản PSO với (20 hạt, 30 vòng) và (30 hạt,
50 vòng). Cột lich_su lưu gbest sau khởi tạo và sau mỗi vòng PSO, nối bằng ';'. Ghi <ket-qua>/E7/e7.csv; chạy lại chỉ
chạy phần thiếu. Phân tích ghi <ket-qua>/bang/e7-dat-j.tex, <ket-qua>/so-lieu-e7.tex và <hinh>/e7-quy-dao-gbest.pdf.
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

from cach_ket_hop import fcm_lap_lai  # noqa: E402
from chay_thuc_nghiem import KET_QUA_MAC_DINH, SEED_DAU, SO_SEED, nap  # noqa: E402
from danh_gia import danh_gia, mau_silhouette  # noqa: E402
from giao_dien import fcm_kpp, fcm_mot_lan  # noqa: E402
from phan_tich import (GOC_TL, TEN_BO, TOL_J, _bang_float, _dau_phay, _plt, ghi_so_lieu, so_viet, ten_macro,  # noqa: E402
                       tex, wilson, mcnemar_cap, holm, macro_doi_chung)
from pso_khoi_tao import PHIEN_BAN, fcm_diem_ngau_nhien, ham_e5  # noqa: E402

CAU_HINH = [("taiwan_kiem_chung", 1.5, 7), ("viet_nam", 1.5, 8), ("taiwan_kiem_chung", 2.0, 3), ("viet_nam", 2.0, 3)]
LUOI = ((20, 30), (30, 50))
HAM = {"fcm": fcm_mot_lan, "fcm_diem": fcm_diem_ngau_nhien, "fcm_kpp": fcm_kpp, "fcm_kpp_20": fcm_lap_lai,
       **{f"pso_{pb}_n{n}_t{t}": ham_e5(pb, n, t) for pb in PHIEN_BAN for n, t in LUOI}}
KHOA = ["bo", "m", "c", "seed"]


def lam(tac_vu):
    bo, m, c, seed = tac_vu
    d = nap(bo)
    mau = mau_silhouette(len(d.X))
    dong = []
    for ten, ham in HAM.items():
        kq = ham(d.X, c, m, np.random.default_rng(seed), None)
        tt = kq.thong_tin
        dong.append(dict(bo=bo, m=m, c=c, seed=seed, phuong_phap=ten, phien_ban=tt.get("phien_ban", ten),
                         n_hat=tt.get("n_hat", 0), so_vong=tt.get("so_vong", 0), j_pso=tt.get("j_pso", np.nan),
                         lich_su=";".join(f"{v:.10g}" for v in kq.lich_su), **danh_gia(d.X, d.y, kq, m, mau)))
    return dong


def chay(thu_muc: str, so_tien_trinh: int) -> int:
    tep = os.path.join(thu_muc, "E7", "e7.csv")
    os.makedirs(os.path.dirname(tep), exist_ok=True)
    cu = pd.read_csv(tep) if os.path.exists(tep) else None
    xong = set() if cu is None else set(zip(cu["bo"], cu["m"], cu["c"], cu["seed"]))
    viec = [(b, m, c, SEED_DAU + s) for b, m, c in CAU_HINH for s in range(SO_SEED) if (b, m, c, SEED_DAU + s) not in xong]
    if viec:
        with Pool(so_tien_trinh) as p:
            for i, ds in enumerate(p.imap_unordered(lam, viec), 1):
                moi = pd.DataFrame(ds)
                cu = moi if cu is None else pd.concat([cu, moi], ignore_index=True)
                cu.sort_values(KHOA + ["phuong_phap"]).to_csv(tep, index=False)
                print(f"[{i}/{len(viec)}] {ds[0]['bo']} m={ds[0]['m']} c={ds[0]['c']} {ds[0]['seed']}", flush=True)
    return len(viec)


TEN_PB = {"goc": "PSO gốc", "w": "PSO-W", "s": "PSO-S", "log": "PSO-LOG", "qpso": "QPSO"}
TEN_DC = {"fcm": "FCM (U ngẫu nhiên)", "fcm_diem": "FCM (điểm ngẫu nhiên)", "fcm_kpp": "FCM (k-means++)",
          "fcm_kpp_20": "FCM (k-means++, 20 lần)"}
NHIEU_CUC_TIEU = [("taiwan_kiem_chung", 1.5, 7), ("viet_nam", 1.5, 8)]
MAU_PB = {"goc": ("#000000", "o", "-"), "w": ("#0072B2", "s", "-"), "s": ("#E69F00", "^", "--"),
          "log": ("#009E73", "D", "-."), "qpso": ("#CC79A7", "v", ":")}


def doc(thu_muc: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(thu_muc, "E7", "e7.csv"))
    j_tot = d.groupby(["bo", "m", "c"])["J_m"].transform("min")
    d["dat_j"] = d["J_m"] <= (1 + TOL_J) * j_tot
    d["j_tot"] = j_tot
    return d


def _ten_ch(b, m, c) -> str:
    return f"{TEN_BO.get(b, b)}, m = {so_viet(m, 1)}, c = {c}"


def bang_dat(d: pd.DataFrame) -> str:
    hang = list(TEN_DC.items()) + [(f"pso_{pb}_n{n}_t{t}", f"{TEN_PB[pb]} ({n} hạt, {t} vòng)")
                                   for n, t in LUOI for pb in PHIEN_BAN]
    than = []
    for pp, ten in hang:
        o = []
        for b, m, c in NHIEU_CUC_TIEU:
            x = d[(d["bo"] == b) & (d["m"] == m) & (d["c"] == c) & (d["phuong_phap"] == pp)]
            k, n = int(x["dat_j"].sum()), len(x)
            lo, hi = wilson(k, n)
            o += [f"{k}/{n}", f"[{so_viet(lo, 0)}; {so_viet(hi, 0)}]", so_viet(x["t_tong"].median(), 2)]
        than.append(f"{tex(ten)} & " + " & ".join(o) + " \\\\")
        if pp == "fcm_kpp_20" or pp == f"pso_qpso_n{LUOI[0][0]}_t{LUOI[0][1]}":
            than.append("\\midrule")
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{tex(_ten_ch(*ch))}}}" for ch in NHIEU_CUC_TIEU) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(NHIEU_CUC_TIEU))),
               "Phương pháp" + " & Đạt & KTC 95\\% & Giây" * len(NHIEU_CUC_TIEU) + " \\\\"]
    return _bang_float("@{}l" + "rrr" * len(NHIEU_CUC_TIEU) + "@{}", tieu_de, than,
                       "TN1 -- số lượt đạt J\\textsubscript{m} tốt nhất trên 2 cấu hình có nhiều cực tiểu",
                       "tab:e7-dat-j",
                       "30 seed mỗi phương pháp; đạt: J\\textsubscript{m} $\\le (1 + 10^{-4})$ lần mức thấp nhất của cấu hình; "
                       "KTC 95\\%: khoảng Wilson của tỉ lệ đạt (\\%); giây: thời gian thực trung vị.")


def quy_dao(d: pd.DataFrame, b: str, m: float, c: int, n: int, t: int) -> dict[str, np.ndarray]:
    """Độ lệch (%) của gbest so với J tốt nhất của cấu hình theo vòng, mỗi phiên bản một ma trận (seed × vòng)."""
    ra = {}
    for pb in PHIEN_BAN:
        x = d[(d["bo"] == b) & (d["m"] == m) & (d["c"] == c) & (d["phuong_phap"] == f"pso_{pb}_n{n}_t{t}")]
        ls = np.array([[float(v) for v in str(r).split(";")] for r in x["lich_su"]])
        ra[pb] = 100 * (ls - x["j_tot"].to_numpy()[:, None]) / x["j_tot"].to_numpy()[:, None]
    return ra


def vong_cai_thien_cuoi(ls: np.ndarray) -> np.ndarray:
    """Chỉ số vòng cuối cùng mà gbest còn giảm, mỗi seed một số (0 nếu không giảm sau khởi tạo)."""
    giam = np.diff(ls, axis=1) < -1e-12
    return np.array([np.flatnonzero(g).max() + 1 if g.any() else 0 for g in giam])


def hinh_quy_dao(d: pd.DataFrame, duong_dan: str, n: int = 30, t: int = 50) -> None:
    plt = _plt()
    cau_hinh = [("viet_nam", 2.0, 3), ("taiwan_kiem_chung", 2.0, 3)]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for nhan, ax, (b, m, c) in zip("ab", axs, cau_hinh):
        qd = quy_dao(d, b, m, c, n, t)
        vong = np.arange(t + 1)
        for pb in PHIEN_BAN:
            mau, dau, net = MAU_PB[pb]
            q1, tv, q3 = np.percentile(qd[pb], [25, 50, 75], axis=0)
            ax.fill_between(vong, q1, q3, color=mau, alpha=0.12, lw=0)
            ax.plot(vong, tv, color=mau, ls=net, lw=1.1, marker=dau, ms=3, markevery=10, label=TEN_PB[pb])
        ax.set_title(f"({nhan}) {_ten_ch(b, m, c)}", fontsize=8)
        ax.set_xlabel("Vòng PSO")
        ax.grid(alpha=0.3)
        _dau_phay(ax, "y")
    axs[0].set_ylabel("Lệch J tốt nhất (%)")
    h, nh = axs[0].get_legend_handles_labels()
    fig.legend(h, nh, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(duong_dan)
    plt.close(fig)


def macro(d: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuEBaySoLuot", so_viet(len(d), 0))]
    for b, m, c in NHIEU_CUC_TIEU:
        goc = ["EBay", b, "C", c]
        for pp in ["fcm", "fcm_diem", "fcm_kpp", "fcm_kpp_20"] + [f"pso_{pb}_n{n}_t{t}" for pb in PHIEN_BAN for n, t in LUOI]:
            x = d[(d["bo"] == b) & (d["m"] == m) & (d["c"] == c) & (d["phuong_phap"] == pp)]
            lo, hi = wilson(int(x["dat_j"].sum()), len(x))
            ra += [("soLieu" + ten_macro(*goc, pp, "K"), str(int(x["dat_j"].sum()))),
                   ("soLieu" + ten_macro(*goc, pp, "WilsonDuoi"), so_viet(lo, 0)),
                   ("soLieu" + ten_macro(*goc, pp, "WilsonTren"), so_viet(hi, 0)),
                   ("soLieu" + ten_macro(*goc, pp, "ThoiGian"), so_viet(x["t_tong"].median(), 2))]
        dc = d[(d["bo"] == b) & (d["m"] == m) & (d["c"] == c)].set_index("seed")
        kpp_s = dc.loc[dc["phuong_phap"] == "fcm_kpp", "dat_j"]
        mc = [mcnemar_cap(g["dat_j"], kpp_s) for _, g in dc[dc["phuong_phap"].str.startswith("pso_")].groupby("phuong_phap")]
        ph = holm([x[2] for x in mc])
        ra += [("soLieu" + ten_macro(*goc, "McSoPhep"), str(len(mc))),
               ("soLieu" + ten_macro(*goc, "McCoYNghia"), str(int((ph < 0.05).sum()))),
               ("soLieu" + ten_macro(*goc, "McPHolmMin"), so_viet(ph.min(), 2))]
        ra += macro_doi_chung(goc, dc)
        pso = d[(d["bo"] == b) & (d["m"] == m) & (d["c"] == c) & d["phuong_phap"].str.startswith("pso_")]
        ra += [("soLieu" + ten_macro(*goc, "PsoKMin"), str(int(pso.groupby("phuong_phap")["dat_j"].sum().min()))),
               ("soLieu" + ten_macro(*goc, "PsoKMax"), str(int(pso.groupby("phuong_phap")["dat_j"].sum().max())))]
    for b in ("viet_nam", "taiwan_kiem_chung"):
        for n, t in LUOI:
            qd = quy_dao(d, b, 2.0, 3, n, t)
            for pb in ("goc", "w", "qpso"):
                goc = ["EBay", b, pb, "Hat", n, "Vong", t]
                vc = vong_cai_thien_cuoi(qd[pb])
                ra += [("soLieu" + ten_macro(*goc, "VongCuoiTv"), so_viet(np.median(vc), 0)),
                       ("soLieu" + ten_macro(*goc, "LechDau"), so_viet(np.median(qd[pb][:, 0]), 1)),
                       ("soLieu" + ten_macro(*goc, "LechVongNam"), so_viet(np.median(qd[pb][:, 5]), 1)),
                       ("soLieu" + ten_macro(*goc, "LechCuoi"), so_viet(np.median(qd[pb][:, -1]), 1))]
    return ra


def phan_tich(thu_muc: str, hinh: str) -> None:
    d = doc(thu_muc)
    os.makedirs(os.path.join(thu_muc, "bang"), exist_ok=True)
    with open(os.path.join(thu_muc, "bang", "e7-dat-j.tex"), "w", encoding="utf-8") as f:
        f.write(bang_dat(d))
    hinh_quy_dao(d, os.path.join(hinh, "e7-quy-dao-gbest.pdf"))
    ghi_so_lieu(macro(d), os.path.join(thu_muc, "so-lieu-e7.tex"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tien-trinh", type=int, default=12)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--hinh", default=os.path.join(GOC_TL, "hinh"))
    ap.add_argument("--chi-phan-tich", action="store_true")
    a = ap.parse_args(argv)
    if not a.chi_phan_tich:
        t0 = time.perf_counter()
        n = chay(os.path.abspath(a.ket_qua), a.tien_trinh)
        print(f"E7: {n} tác vụ, {time.perf_counter() - t0:.0f} s", flush=True)
    phan_tich(os.path.abspath(a.ket_qua), os.path.abspath(a.hinh))


if __name__ == "__main__":
    main()
