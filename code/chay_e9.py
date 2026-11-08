# -*- coding: utf-8 -*-
"""Thí nghiệm E9: 5 phiên bản PSO (gốc, W, S, LOG, QPSO) trong vai trò tìm trọng số thuộc tính cho FCM.

    python3 chay_e9.py [--tien-trinh 12] [--ket-qua DIR] [--hinh DIR] [--chi-phan-tich]

Hai bộ chính (Việt Nam, Taiwan Default) × c ∈ {2, 3, 4} × 30 seed, m = 2. Hai ngân sách: 500 lần tính fitness (10 hạt) và
1.000 lần (20 hạt), cùng 49 vòng sau khởi tạo. Mỗi (bộ, c, seed) chạy tuần tự FCM k-means++, tìm ngẫu nhiên ở 2 ngân
sách và 5 phiên bản × 2 ngân sách (pso_trong_so), mỗi phương pháp nhận default_rng(seed) riêng nên cùng V0, cùng mẫu
Silhouette. Ghi <ket-qua>/E9/e9.csv (chạy lại chỉ chạy phần thiếu).

Kiểm định (chốt trước khi chạy): ở mỗi (bộ, c, ngân sách), mỗi bản cải tiến so với PSO gốc và mỗi phiên bản so với tìm
ngẫu nhiên cùng ngân sách, trên −fitness (Silhouette − phạt), auc_cung và chenh_ti_le; Wilcoxon theo cặp seed, Holm trên
toàn họ. Phân tích ghi <ket-qua>/bang/e9-*.tex, <ket-qua>/so-lieu-e9.tex và <hinh>/e9-quy-dao.pdf.
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

from chay_thuc_nghiem import KET_QUA_MAC_DINH, SEED_DAU, SO_SEED, nap  # noqa: E402
from danh_gia import danh_gia, mau_silhouette  # noqa: E402
from giao_dien import fcm_kpp  # noqa: E402
from phan_tich import (GOC_TL, DAU_KL, TEN_BO, _bang_float, _dau_phay, _plt, ghi_so_lieu, holm, so_sanh_cap,  # noqa: E402
                       so_viet, ten_macro, tex)
from pso_khoi_tao import PHIEN_BAN  # noqa: E402
from pso_trong_so import ngau_nhien_trong_so, pso_trong_so  # noqa: E402

BO = ["viet_nam", "taiwan_kiem_chung"]
SO_CUM = (2, 3, 4)
M = 2.0
NGAN_SACH = {500: 10, 1000: 20}  # ngân sách → số hạt; số vòng = ngân sách / số hạt − 1 = 49
KHOA = ["bo", "c", "seed"]


def _pso(pb: str, ns: int):
    def fit(X, c, m, rng, max_fe=None):
        return pso_trong_so(X, c, m, rng, ns, phien_ban=pb, n_hat=NGAN_SACH[ns])
    return fit


def _ngau_nhien(ns: int):
    def fit(X, c, m, rng, max_fe=None):
        return ngau_nhien_trong_so(X, c, m, rng, ns)
    return fit


HAM = {"fcm_kpp": fcm_kpp, **{f"ngau_nhien_{ns}": _ngau_nhien(ns) for ns in NGAN_SACH},
       **{f"pso_{pb}_{ns}": _pso(pb, ns) for ns in NGAN_SACH for pb in PHIEN_BAN}}


def lam(tac_vu):
    bo, c, seed = tac_vu
    d = nap(bo)
    mau = mau_silhouette(len(d.X))
    dong = []
    for ten, ham in HAM.items():
        kq = ham(d.X, c, M, np.random.default_rng(seed), None)
        w = kq.thong_tin.get("trong_so")
        dong.append(dict(bo=bo, c=c, seed=seed, phuong_phap=ten, fitness=kq.thong_tin.get("fitness", np.nan),
                         trong_so="" if w is None else ";".join(f"{v:.6g}" for v in w),
                         lich_su=";".join(f"{v:.10g}" for v in kq.lich_su), **danh_gia(d.X, d.y, kq, M, mau)))
    return dong


def chay(thu_muc: str, so_tien_trinh: int) -> int:
    tep = os.path.join(thu_muc, "E9", "e9.csv")
    os.makedirs(os.path.dirname(tep), exist_ok=True)
    cu = pd.read_csv(tep) if os.path.exists(tep) else None
    xong = set() if cu is None else set(zip(cu["bo"], cu["c"], cu["seed"]))
    viec = [(b, c, SEED_DAU + s) for b in BO for c in sorted(SO_CUM, reverse=True) for s in range(SO_SEED)
            if (b, c, SEED_DAU + s) not in xong]
    if viec:
        with Pool(so_tien_trinh) as p:
            for i, ds in enumerate(p.imap_unordered(lam, viec), 1):
                moi = pd.DataFrame(ds)
                cu = moi if cu is None else pd.concat([cu, moi], ignore_index=True)
                cu.sort_values(KHOA + ["phuong_phap"]).to_csv(tep, index=False)
                print(f"[{i}/{len(viec)}] {ds[0]['bo']} c={ds[0]['c']} {ds[0]['seed']}", flush=True)
    return len(viec)


# ------------------------------------------------------------------ phân tích
TEN_PB = {"goc": "Gốc", "w": "W", "s": "S", "log": "LOG", "qpso": "QPSO"}
MAU_PB = {"goc": ("#000000", "-"), "w": ("#0072B2", "-"), "s": ("#E69F00", "--"), "log": ("#009E73", "-."),
          "qpso": ("#CC79A7", ":")}
CHI_SO = ["diem", "auc_cung", "chenh_ti_le"]  # diem = −fitness = Silhouette − phạt (lớn hơn là tốt hơn)
TEN_CS = {"diem": "Silhouette − phạt", "auc_cung": "AUC phân hoạch", "chenh_ti_le": "Chênh tỉ lệ vỡ nợ"}


def doc(thu_muc: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(thu_muc, "E9", "e9.csv"))
    d["diem"] = -d["fitness"]
    d["ngan_sach"] = d["phuong_phap"].str.extract(r"_(\d+)$")[0].astype(float)
    d["phien_ban"] = d["phuong_phap"].str.extract(r"^pso_([a-z]+)_")[0]
    return d


def kiem_dinh(d: pd.DataFrame) -> pd.DataFrame:
    """Hai họ trong cùng một hiệu chỉnh Holm: bản cải tiến so với gốc, và mỗi phiên bản so với tìm ngẫu nhiên."""
    phan = []
    for ns in NGAN_SACH:
        x = d[d["ngan_sach"] == ns].copy()
        for pb in PHIEN_BAN:
            doi = ([f"pso_goc_{ns}"] if pb != "goc" else []) + [f"ngau_nhien_{ns}"]
            k = so_sanh_cap(x, ["bo", "c"], f"pso_{pb}_{ns}", doi, CHI_SO)
            k["ngan_sach"], k["phien_ban"] = ns, pb
            phan.append(k)
    kd = pd.concat(phan, ignore_index=True)
    kd["p_holm"] = holm(kd["p"].to_numpy())
    kd["ket_luan"] = np.where(kd["p_holm"] < 0.05, np.sign(kd["hieu_tv"]), 0).astype(int)
    kd["so_voi"] = np.where(kd["doi_chung"].str.startswith("pso_goc"), "goc", "ngau_nhien")
    return kd


def _kl(kd, b, c, ns, pb, so_voi, cs) -> int:
    k = kd[(kd["bo"] == b) & (kd["c"] == c) & (kd["ngan_sach"] == ns) & (kd["phien_ban"] == pb)
           & (kd["so_voi"] == so_voi) & (kd["chi_so"] == cs)]
    return int(k["ket_luan"].iloc[0]) if len(k) else 0


def bang_chi_so(d: pd.DataFrame, kd: pd.DataFrame, cs: str, nhan: str) -> str:
    tv = d.groupby(["bo", "c", "phuong_phap"])[cs].median()
    than = []
    for b in BO:
        for i, c in enumerate(SO_CUM):
            for j, ns in enumerate(NGAN_SACH):
                cot = [f"ngau_nhien_{ns}"] + [f"pso_{pb}_{ns}" for pb in PHIEN_BAN]
                cao = max(tv.loc[(b, c, pp)] for pp in cot)
                o = [tex(TEN_BO[b]) if i == j == 0 else "", str(c) if j == 0 else "", so_viet(ns, 0),
                     so_viet(tv.loc[(b, c, "fcm_kpp")], 3) if j == 0 else ""]
                for pp in cot:
                    s = so_viet(tv.loc[(b, c, pp)], 3)
                    if pp.startswith("pso_"):
                        pb = pp.split("_")[1]
                        s += DAU_KL[_kl(kd, b, c, ns, pb, "ngau_nhien", cs)]
                        if pb != "goc" and _kl(kd, b, c, ns, pb, "goc", cs) != 0:
                            s += "\\textsuperscript{" + ("*" if _kl(kd, b, c, ns, pb, "goc", cs) > 0 else "\\dag") + "}"
                    o.append(f"\\textbf{{{s}}}" if tv.loc[(b, c, pp)] >= cao - 5e-4 else s)
                than.append(" & ".join(o) + " \\\\")
        if b != BO[-1]:
            than.append("\\midrule")
    tieu_de = ["Bộ dữ liệu & c & Ngân sách & FCM & NN & " + " & ".join(TEN_PB[pb] for pb in PHIEN_BAN) + " \\\\"]
    return _bang_float("@{}lllrrrrrrr@{}", tieu_de, [r for r in than],
                       f"TN4 -- {TEN_CS[cs]} của 5 phiên bản PSO khi tìm trọng số thuộc tính", nhan,
                       "Trung vị 30 seed, m = 2; ngân sách là số lần tính fitness (10 hạt với 500, 20 hạt với 1.000). "
                       "NN: tìm ngẫu nhiên cùng ngân sách. Đậm: cao nhất trong các phương pháp tìm kiếm cùng ngân sách. "
                       "\\textsuperscript{+} / \\textsuperscript{--}: tốt hơn / kém hơn NN có ý nghĩa; "
                       "\\textsuperscript{*} / \\textsuperscript{\\dag}: tốt hơn / kém hơn PSO gốc có ý nghĩa "
                       "(Wilcoxon theo cặp seed, Holm toàn họ, $\\alpha = 0{,}05$).")


def bang_tong_hop(kd: pd.DataFrame) -> str:
    than = []
    for pb in PHIEN_BAN:
        o = [TEN_PB[pb]]
        for so_voi in ("ngau_nhien", "goc"):
            for cs in CHI_SO:
                k = kd[(kd["phien_ban"] == pb) & (kd["so_voi"] == so_voi) & (kd["chi_so"] == cs)]["ket_luan"]
                o.append("--" if k.empty else f"{(k == 1).sum()}/{(k == 0).sum()}/{(k == -1).sum()}")
        than.append(" & ".join(o) + " \\\\")
    ten = " & ".join(["Điểm", "AUC", "$\\Delta r$"])
    tieu_de = ["& \\multicolumn{3}{c}{So với tìm ngẫu nhiên} & \\multicolumn{3}{c}{So với PSO gốc} \\\\",
               "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}", f"Phiên bản & {ten} & {ten} \\\\"]
    so_o = kd.groupby(["bo", "c", "ngan_sach"]).ngroups
    return _bang_float("@{}lcccccc@{}", tieu_de, than,
                       "TN4 -- số ô mỗi phiên bản tốt hơn / không khác / kém hơn tìm ngẫu nhiên và PSO gốc",
                       "tab:e9-tong-hop",
                       f"{so_o} ô (2 bộ × 3 số cụm × 2 ngân sách). Điểm: Silhouette − phạt (chính là fitness); "
                       "Wilcoxon theo cặp seed, Holm toàn họ, $\\alpha = 0{,}05$.")


def hinh_quy_dao(d: pd.DataFrame, duong_dan: str, c: int = 3, ns: int = 1000) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for nhan, ax, b in zip("ab", axs, BO):
        for pp, ten, (mau, net) in ([(f"pso_{pb}_{ns}", TEN_PB[pb], MAU_PB[pb]) for pb in PHIEN_BAN]
                                   + [(f"ngau_nhien_{ns}", "NN", ("#999999", "--"))]):
            x = d[(d["bo"] == b) & (d["c"] == c) & (d["phuong_phap"] == pp)]
            ls = -np.array([[float(v) for v in s.split(";")] for s in x["lich_su"]])
            n_hat = NGAN_SACH[ns] if pp.startswith("pso_") else 10
            fe = n_hat * np.arange(1, ls.shape[1] + 1)
            ax.plot(fe, np.median(ls, axis=0), color=mau, ls=net, lw=1.2, label=ten)
        ax.set_title(f"({nhan}) {TEN_BO[b]}, c = {c}, {so_viet(ns, 0)} lần tính fitness", fontsize=8)
        ax.set_xlabel("Số lần tính fitness")
        ax.grid(alpha=0.3)
        _dau_phay(ax, "y")
    axs[0].set_ylabel("Silhouette − phạt (trung vị)")
    h, nh = axs[0].get_legend_handles_labels()
    fig.legend(h, nh, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(duong_dan)
    plt.close(fig)


def macro(d: pd.DataFrame, kd: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuEChinSoLuot", so_viet(len(d), 0)), ("soLieuEChinSoO", str(kd.groupby(["bo", "c", "ngan_sach"]).ngroups)),
          ("soLieuEChinSoPhep", str(len(kd)))]
    for pb in PHIEN_BAN:
        for so_voi in ("ngau_nhien", "goc"):
            for cs in CHI_SO:
                k = kd[(kd["phien_ban"] == pb) & (kd["so_voi"] == so_voi) & (kd["chi_so"] == cs)]["ket_luan"]
                for ten, v in (("Hon", 1), ("Bang", 0), ("Kem", -1)):
                    ra.append(("soLieu" + ten_macro("EChin", pb, so_voi, cs, ten), str(int((k == v).sum()))))
    tv = d.groupby(["bo", "c", "phuong_phap"])[CHI_SO + ["t_tong"]].median()
    for (b, c, pp), r in tv.iterrows():
        for cs in CHI_SO + ["t_tong"]:
            ra.append(("soLieu" + ten_macro("EChin", b, "C", c, pp, cs), so_viet(r[cs], 1 if cs == "t_tong" else 3)))
    return ra


def phan_tich(thu_muc: str, hinh: str) -> None:
    d = doc(thu_muc)
    kd = kiem_dinh(d)
    kd.to_csv(os.path.join(thu_muc, "E9", "kiem-dinh.csv"), index=False)
    os.makedirs(os.path.join(thu_muc, "bang"), exist_ok=True)
    bang = {"e9-diem": bang_chi_so(d, kd, "diem", "tab:e9-diem"),
            "e9-auc": bang_chi_so(d, kd, "auc_cung", "tab:e9-auc"),
            "e9-chenh": bang_chi_so(d, kd, "chenh_ti_le", "tab:e9-chenh"),
            "e9-tong-hop": bang_tong_hop(kd)}
    for ten, noi_dung in bang.items():
        with open(os.path.join(thu_muc, "bang", f"{ten}.tex"), "w", encoding="utf-8") as f:
            f.write(noi_dung)
    hinh_quy_dao(d, os.path.join(hinh, "e9-quy-dao.pdf"))
    ghi_so_lieu(macro(d, kd), os.path.join(thu_muc, "so-lieu-e9.tex"))


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
        print(f"E9: {n} tác vụ, {time.perf_counter() - t0:.0f} s", flush=True)
    phan_tich(os.path.abspath(a.ket_qua), os.path.abspath(a.hinh))


if __name__ == "__main__":
    main()
