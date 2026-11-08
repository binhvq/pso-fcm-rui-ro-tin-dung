# -*- coding: utf-8 -*-
"""Thí nghiệm E8: PSO tìm trọng số thuộc tính cho FCM trên các bộ dữ liệu tín dụng.

    python3 chay_e8.py [--tien-trinh 16] [--ket-qua DIR] [--hinh DIR] [--chi-phan-tich]

Bảy bộ tín dụng × c ∈ {2, 3, 4} × 30 seed. Mỗi (bộ, c, seed) chạy tuần tự 4 phương pháp, mỗi phương pháp nhận
default_rng(seed) riêng: FCM k-means++, WFCM trọng số giải tích (không PSO), tìm ngẫu nhiên trọng số và PSO tìm trọng
số (cùng ngân sách 500 lần chấm, cùng V0 và mẫu Silhouette). Ghi <ket-qua>/E8/e8.csv (chạy lại chỉ chạy phần thiếu).

Chỉ số chính, chốt trước khi chạy: auc_cung (AUC của chính phân hoạch cứng) và chenh_ti_le (chênh tỉ lệ vỡ nợ giữa
các cụm) — đều dùng nhãn, nhãn không đi vào fitness. Silhouette là chỉ số phụ vì chính nó là fitness của PSO.
Kiểm định: PSO so với từng đối chứng, Wilcoxon theo cặp seed ở từng (bộ, c, chỉ số), Holm trên toàn họ phép thử.
Phân tích ghi <ket-qua>/bang/e8-*.tex, <ket-qua>/so-lieu-e8.tex và <hinh>/e8-*.pdf.
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

from cach_ket_hop import wfcm_giai_tich  # noqa: E402
from chay_thuc_nghiem import KET_QUA_MAC_DINH, SEED_DAU, SO_SEED, nap  # noqa: E402
from danh_gia import danh_gia, mau_silhouette  # noqa: E402
from giao_dien import fcm_kpp  # noqa: E402
from phan_tich import (GOC_TL, TEN_BO, _bang_float, _plt, ghi_so_lieu, holm, so_viet, ten_macro, tex,  # noqa: E402
                       wilcoxon_cap)
from pso_trong_so import N_HAT, ngau_nhien_trong_so, pso_trong_so  # noqa: E402

BO = ["viet_nam", "taiwan_kiem_chung", "polish", "taiwan_pha_san", "south_german", "german_so", "australian"]
SO_CUM = (2, 3, 4)
M = 2.0
HAM = {"fcm_kpp": fcm_kpp, "wfcm_giai_tich": wfcm_giai_tich, "ngau_nhien_trong_so": ngau_nhien_trong_so,
       "pso_trong_so": pso_trong_so}
KHOA = ["bo", "c", "seed"]


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
    tep = os.path.join(thu_muc, "E8", "e8.csv")
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
TEN_PP = {"fcm_kpp": "FCM", "wfcm_giai_tich": "WFCM giải tích", "ngau_nhien_trong_so": "Ngẫu nhiên",
          "pso_trong_so": "PSO"}
DOI_CHUNG = ["fcm_kpp", "wfcm_giai_tich", "ngau_nhien_trong_so"]
CHI_SO = {"auc_cung": "AUC phân hoạch", "chenh_ti_le": "Chênh tỉ lệ vỡ nợ", "silhouette": "Silhouette"}
ALPHA = 0.05


def doc(thu_muc: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(thu_muc, "E8", "e8.csv"))


def kiem_dinh(d: pd.DataFrame) -> pd.DataFrame:
    """Mỗi (bộ, c, chỉ số, đối chứng): trung vị hiệu PSO − đối chứng theo cặp seed, p Wilcoxon, p Holm toàn họ và
    kết luận +1 (PSO tốt hơn có ý nghĩa), −1 (kém hơn có ý nghĩa), 0 (không khác)."""
    hang = []
    for (b, c), g in d.groupby(["bo", "c"]):
        p = g.pivot(index="seed", columns="phuong_phap")
        for cs in CHI_SO:
            for dc in DOI_CHUNG:
                hieu = (p[cs]["pso_trong_so"] - p[cs][dc]).dropna()
                hang.append(dict(bo=b, c=c, chi_so=cs, doi_chung=dc, hieu_tv=float(hieu.median()),
                                 p=wilcoxon_cap(hieu.to_numpy(), np.zeros(len(hieu)))))
    kd = pd.DataFrame(hang)
    kd["p_holm"] = holm(kd["p"].to_numpy())
    kd["ket_luan"] = np.where(kd["p_holm"] < ALPHA, np.sign(kd["hieu_tv"]), 0).astype(int)
    return kd


def _ten_bo(b: str) -> str:
    return TEN_BO.get(b, b)


def bang_ket_qua(d: pd.DataFrame, kd: pd.DataFrame) -> str:
    tv = d.groupby(["bo", "c", "phuong_phap"])[list(CHI_SO)].median()
    than = []
    for b in BO:
        for i, c in enumerate(SO_CUM):
            o = [tex(_ten_bo(b)) if i == 0 else "", str(c)]
            for cs in ("auc_cung", "chenh_ti_le"):
                cao = max(tv.loc[(b, c, pp), cs] for pp in HAM)
                for pp in HAM:
                    s = so_viet(tv.loc[(b, c, pp), cs], 3)
                    if pp != "pso_trong_so":
                        k = kd[(kd["bo"] == b) & (kd["c"] == c) & (kd["chi_so"] == cs) & (kd["doi_chung"] == pp)]
                        s += {1: "\\textsuperscript{+}", -1: "\\textsuperscript{--}", 0: ""}[int(k["ket_luan"].iloc[0])]
                    o.append(f"\\textbf{{{s}}}" if tv.loc[(b, c, pp), cs] >= cao - 5e-4 else s)
            than.append(" & ".join(o) + " \\\\")
        if b != BO[-1]:
            than.append("\\midrule")
    ngan = {"fcm_kpp": "FCM", "wfcm_giai_tich": "WFCM", "ngau_nhien_trong_so": "NN", "pso_trong_so": "PSO"}
    ten = " & ".join(ngan[pp] for pp in HAM)
    tieu_de = ["& & \\multicolumn{4}{c}{AUC phân hoạch} & \\multicolumn{4}{c}{Chênh tỉ lệ vỡ nợ} \\\\",
               "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
               f"Bộ dữ liệu & c & {ten} & {ten} \\\\"]
    return _bang_float("@{}llrrrrrrrr@{}", tieu_de, than,
                       "TN3 -- PSO tìm trọng số thuộc tính so với 3 đối chứng trên 7 bộ dữ liệu tín dụng",
                       "tab:e8-ket-qua",
                       "Trung vị 30 seed, m = 2. FCM: FCM k-means++; WFCM: trọng số giải tích; NN: tìm ngẫu nhiên trọng số. Đậm: cao nhất hàng. Dấu \\textsuperscript{+} / \\textsuperscript{--} "
                       "sau một đối chứng: PSO tốt hơn / kém hơn đối chứng đó có ý nghĩa (Wilcoxon theo cặp seed, "
                       "Holm toàn họ, $\\alpha = 0{,}05$); không dấu: không khác có ý nghĩa.")


def bang_tong_hop(d: pd.DataFrame, kd: pd.DataFrame) -> str:
    than = []
    for cs, ten_cs in CHI_SO.items():
        o = [tex(ten_cs)]
        for dc in DOI_CHUNG:
            k = kd[(kd["chi_so"] == cs) & (kd["doi_chung"] == dc)]["ket_luan"]
            o.append(f"{(k == 1).sum()} / {(k == 0).sum()} / {(k == -1).sum()}")
        than.append(" & ".join(o) + " \\\\")
    tg = d.groupby("phuong_phap")["t_tong"].median()
    than += ["\\midrule", "Thời gian trung vị (giây) & " + " & ".join(so_viet(tg[dc], 2) for dc in DOI_CHUNG)
             + f" \\\\", "Thời gian PSO (giây) & \\multicolumn{3}{c}{" + so_viet(tg["pso_trong_so"], 1) + "} \\\\"]
    tieu_de = ["PSO so với & " + " & ".join(tex(TEN_PP[dc]) for dc in DOI_CHUNG) + " \\\\"]
    so_o = d.groupby(["bo", "c"]).ngroups
    return _bang_float("@{}lccc@{}", tieu_de, than,
                       "TN3 -- số cấu hình PSO tìm trọng số tốt hơn / không khác / kém hơn từng đối chứng",
                       "tab:e8-tong-hop",
                       f"{so_o} cấu hình (7 bộ × 3 số cụm), Wilcoxon theo cặp seed, Holm toàn họ, "
                       "$\\alpha = 0{,}05$. Silhouette là fitness của PSO nên chỉ là chỉ số phụ.")


def trong_so_tv(d: pd.DataFrame, b: str, c: int) -> pd.Series:
    x = d[(d["bo"] == b) & (d["c"] == c) & (d["phuong_phap"] == "pso_trong_so")]
    W = np.array([[float(v) for v in s.split(";")] for s in x["trong_so"]])
    W = W / W.mean(axis=1, keepdims=True)  # cùng thang với hệ số nhân: trung bình 1
    return pd.Series(np.median(W, axis=0), index=list(nap(b).ten_cot)).sort_values()


def hinh_trong_so(d: pd.DataFrame, duong_dan: str, c: int = 3) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.6))
    for nhan, ax, b in zip("ab", axs, ("viet_nam", "taiwan_kiem_chung")):
        w = trong_so_tv(d, b, c)
        ax.barh(range(len(w)), w.to_numpy(), color="#0072B2", height=0.7)
        ax.set_yticks(range(len(w)), [tex(t).replace("\\_", "_") for t in w.index], fontsize=6)
        ax.axvline(1.0, color="#000000", lw=0.8, ls="--")
        ax.set_title(f"({nhan}) {_ten_bo(b)}, c = {c}", fontsize=8)
        ax.set_xlabel("Trọng số tương đối (trung vị)")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(duong_dan)
    plt.close(fig)


def hinh_hoi_tu(d: pd.DataFrame, duong_dan: str, c: int = 3) -> None:
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for nhan, ax, b in zip("ab", axs, ("viet_nam", "taiwan_kiem_chung")):
        for pp, mau, net in (("pso_trong_so", "#0072B2", "-"), ("ngau_nhien_trong_so", "#E69F00", "--")):
            x = d[(d["bo"] == b) & (d["c"] == c) & (d["phuong_phap"] == pp)]
            ls = -np.array([[float(v) for v in s.split(";")] for s in x["lich_su"]])
            fe = N_HAT * np.arange(1, ls.shape[1] + 1)
            q1, tv, q3 = np.percentile(ls, [25, 50, 75], axis=0)
            ax.fill_between(fe, q1, q3, color=mau, alpha=0.15, lw=0)
            ax.plot(fe, tv, color=mau, ls=net, lw=1.2, label=TEN_PP[pp])
        ax.set_title(f"({nhan}) {_ten_bo(b)}, c = {c}", fontsize=8)
        ax.set_xlabel("Số lần chấm fitness")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Silhouette − phạt (tốt nhất)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(duong_dan)
    plt.close(fig)


def macro(d: pd.DataFrame, kd: pd.DataFrame) -> list[tuple[str, str]]:
    tg = d.groupby("phuong_phap")["t_tong"].median()
    ra = [("soLieuETamThoiGianPso", so_viet(tg["pso_trong_so"], 0)), ("soLieuETamThoiGianFcm", so_viet(tg["fcm_kpp"], 2)),
          ("soLieuETamSoLuot", so_viet(len(d), 0)), ("soLieuETamSoCauHinh", str(d.groupby(["bo", "c"]).ngroups)),
          ("soLieuETamSoPhep", str(len(kd)))]
    for cs in CHI_SO:
        for dc in DOI_CHUNG:
            k = kd[(kd["chi_so"] == cs) & (kd["doi_chung"] == dc)]["ket_luan"]
            for ten, v in (("Hon", 1), ("Bang", 0), ("Kem", -1)):
                ra.append(("soLieu" + ten_macro("ETam", cs, dc, ten), str(int((k == v).sum()))))
    cot = list(CHI_SO) + ["cum_nho_nhat", "t_tong"]
    tv = d.groupby(["bo", "c", "phuong_phap"])[cot].median()
    for (b, c, pp), r in tv.iterrows():
        for cs in cot:
            ra.append(("soLieu" + ten_macro("ETam", b, "C", c, pp, cs), so_viet(r[cs], 2 if cs == "t_tong" else 3)))
    for b in ("viet_nam", "taiwan_kiem_chung"):
        w = trong_so_tv(d, b, 3)
        for i, (ten, v) in enumerate(w.iloc[::-1].head(3).items(), 1):
            ra += [("soLieu" + ten_macro("ETam", b, "TrongSoCao", i, "Ten"), tex(ten)),
                   ("soLieu" + ten_macro("ETam", b, "TrongSoCao", i, "GiaTri"), so_viet(v, 2))]
        ra.append(("soLieu" + ten_macro("ETam", b, "SoThuocTinh"), str(len(w))))
    return ra


def phan_tich(thu_muc: str, hinh: str) -> None:
    d = doc(thu_muc)
    kd = kiem_dinh(d)
    kd.to_csv(os.path.join(thu_muc, "E8", "kiem-dinh.csv"), index=False)
    os.makedirs(os.path.join(thu_muc, "bang"), exist_ok=True)
    for ten, noi_dung in (("e8-ket-qua", bang_ket_qua(d, kd)), ("e8-tong-hop", bang_tong_hop(d, kd))):
        with open(os.path.join(thu_muc, "bang", f"{ten}.tex"), "w", encoding="utf-8") as f:
            f.write(noi_dung)
    hinh_trong_so(d, os.path.join(hinh, "e8-trong-so.pdf"))
    hinh_hoi_tu(d, os.path.join(hinh, "e8-hoi-tu.pdf"))
    ghi_so_lieu(macro(d, kd), os.path.join(thu_muc, "so-lieu-e8.tex"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tien-trinh", type=int, default=16)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--hinh", default=os.path.join(GOC_TL, "hinh"))
    ap.add_argument("--chi-phan-tich", action="store_true")
    a = ap.parse_args(argv)
    if not a.chi_phan_tich:
        t0 = time.perf_counter()
        n = chay(os.path.abspath(a.ket_qua), a.tien_trinh)
        print(f"E8: {n} tác vụ, {time.perf_counter() - t0:.0f} s", flush=True)
    phan_tich(os.path.abspath(a.ket_qua), os.path.abspath(a.hinh))


if __name__ == "__main__":
    main()
