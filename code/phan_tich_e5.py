# -*- coding: utf-8 -*-
"""Phân tích E5 (PSO gốc và các phiên bản cải tiến làm bộ khởi tạo tâm cho FCM): bảng, hình, macro.

    python3 phan_tich_e5.py [--ket-qua DIR] [--hinh DIR]

Đọc <ket-qua>/E5/*.csv, ghi <ket-qua>/bang/e5-*.tex, <ket-qua>/so-lieu-e5.tex và <hinh>/e5-*.pdf.
"Đạt J tốt nhất": J_m ≤ (1 + 10⁻⁴)·J_m nhỏ nhất của mọi lượt, mọi phương pháp trên cùng (bộ, c) — cùng quy tắc với E2.
"Khoảng cách J sau PSO": (J của gbest trước khi FCM chạy − J tốt nhất) / J tốt nhất, tính bằng %.
Kiểm định: mỗi phiên bản cải tiến so với PSO gốc trên khoảng cách J sau PSO, Wilcoxon theo cặp seed ở từng ô
(bộ, c, số hạt, số vòng), Holm trên toàn bộ họ phép thử.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from chay_e5 import SO_HAT, SO_VONG
from phan_tich import (GOC_TL, TEN_BO, TOL_J, _bang_float, _dau_phay, _plt, ghi_so_lieu, hai_dong, holm, o_tv_iqr, so_gon, so_viet, ten_macro, tex, wilcoxon_cap, wilson, mcnemar_cap, macro_doi_chung)
from pso_khoi_tao import PHIEN_BAN

N_MAC_DINH, T_MAC_DINH = 20, 30
TEN_PB = {"goc": "PSO gốc (1995)", "w": "PSO-W (1998)", "s": "PSO-S (2022)", "log": "PSO-LOG (2022)",
          "qpso": "QPSO (2004)"}
DOI_CHUNG = {"fcm": "FCM (U ngẫu nhiên)", "fcm_diem": "FCM (điểm ngẫu nhiên)", "fcm_kpp": "FCM (k-means++)"}
THU_TU_BO = ["viet_nam", "taiwan_kiem_chung", "taiwan_kiem_chung_khong_log"]


def doc(thu_muc: str) -> pd.DataFrame:
    tep = sorted(glob.glob(os.path.join(thu_muc, "E5", "*.csv")))
    if not tep:
        raise FileNotFoundError(f"không có CSV E5 trong {thu_muc}")
    d = pd.concat([pd.read_csv(t) for t in tep], ignore_index=True)
    loi = d["loi"].fillna("").ne("")
    if loi.any():
        raise RuntimeError(f"{int(loi.sum())} dòng E5 có lỗi: {d.loc[loi, 'phuong_phap'].unique()[:5]}")
    j_tot = d.groupby(["bo", "c"])["J_m"].transform("min")
    d["dat_j"] = d["J_m"] <= (1 + TOL_J) * j_tot
    d["khoang_cach_pso"] = 100 * (d["j_pso"] - j_tot) / j_tot
    return d


def _ten_pp(pb: str, n: int = N_MAC_DINH, t: int = T_MAC_DINH) -> str:
    return f"pso_{pb}_n{n}_t{t}"


def _ten_bo(bo: str, c: int) -> str:
    return f"{TEN_BO.get(bo, bo)}, c~=~{c}"


def tom_tat(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["bo", "c", "phuong_phap"])
    return pd.DataFrame(dict(
        dat_j=100 * g["dat_j"].mean(), khoang_cach_pso=g["khoang_cach_pso"].median(), vong_fcm=g["n_vong_fcm"].median(),
        t_tong=g["t_tong"].median(), auc=g["auc_rui_ro"].median(), so_luot=g.size())).reset_index()


def _cau_hinh(d: pd.DataFrame) -> list[tuple[str, int]]:
    ch = {(b, int(c)) for b, c in zip(d["bo"], d["c"])}
    return sorted(ch, key=lambda bc: (THU_TU_BO.index(bc[0]) if bc[0] in THU_TU_BO else 99, bc[1]))


def bang_tong_hop(tt: pd.DataFrame) -> str:
    ch = _cau_hinh(tt)
    hang = [(k, v) for k, v in DOI_CHUNG.items()] + [(_ten_pp(pb), TEN_PB[pb]) for pb in PHIEN_BAN]
    than = []
    for i, (pp, ten) in enumerate(hang):
        if i == len(DOI_CHUNG):
            than.append("\\midrule")
        o = []
        for b, c in ch:
            r = tt[(tt["bo"] == b) & (tt["c"] == c) & (tt["phuong_phap"] == pp)]
            if r.empty:
                o += ["--"] * 3
                continue
            r = r.iloc[0]
            o += [so_viet(r.dat_j, 0), so_viet(r.khoang_cach_pso, 1) if pp != "fcm" and pp != "fcm_kpp" else "--",
                  so_gon(r.t_tong, 2)]
        than.append(f"{tex(ten)} & " + " & ".join(o) + " \\\\")
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{tex(_ten_bo(b, c))}}}" for b, c in ch) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(ch))),
               "Phương pháp" + (" & " + " & ".join([hai_dong("Đạt", "(\\%)"), hai_dong("Lệch", "(\\%)"),
                                                     hai_dong("Thời gian", "(s)")])) * len(ch) + " \\\\"]
    return _bang_float("@{}l" + "rrr" * len(ch) + "@{}", tieu_de, than,
                       f"TN1 -- PSO gốc và các phiên bản cải tiến làm bộ khởi tạo tâm cho FCM "
                       f"({N_MAC_DINH} hạt, {T_MAC_DINH} vòng, 30 seed)",
                       "tab:e5-tong-hop",
                       "Đạt: tỉ lệ lượt đạt J tốt nhất sau khi FCM chạy tới hội tụ; lệch: J\\textsubscript{m} của bộ tâm đưa vào FCM (gbest của PSO, hoặc c điểm ngẫu "
                       "nhiên) so với J\\textsubscript{m} tốt nhất (\\%), trung vị; thời gian gồm PSO và FCM, trung vị. Bầy PSO khởi tạo bằng điểm dữ liệu ngẫu nhiên, chạy đủ "
                       "số vòng, c\\textsubscript{1} = c\\textsubscript{2} = 2 (trừ QPSO).")


def bang_luoi(tt: pd.DataFrame, bo: str, c: int) -> str:
    luoi = [(n, t) for n in SO_HAT for t in SO_VONG]
    d = tt[(tt["bo"] == bo) & (tt["c"] == c)]
    than = []
    for pb in PHIEN_BAN:
        o = []
        for n, t in luoi:
            r = d[d["phuong_phap"] == _ten_pp(pb, n, t)]
            o.append("--" if r.empty else so_viet(r.iloc[0].dat_j, 0))
        than.append(f"{tex(TEN_PB[pb])} & " + " & ".join(o) + " \\\\")
    for pp in DOI_CHUNG:
        r = d[d["phuong_phap"] == pp]
        if not r.empty:
            than.append(f"\\multicolumn{{{1 + len(luoi)}}}{{@{{}}l}}{{\\textit{{Đối chứng {tex(DOI_CHUNG[pp])}: "
                        f"{so_viet(r.iloc[0].dat_j, 0)}\\%}}}} \\\\")
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{{len(SO_VONG)}}}{{c}}{{{n} hạt}}" for n in SO_HAT) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + len(SO_VONG) * i}-{1 + len(SO_VONG) * (i + 1)}}}"
                        for i in range(len(SO_HAT))),
               "Phiên bản & " + " & ".join(f"T = {t}" for _, t in luoi) + " \\\\"]
    return _bang_float("@{}l" + "r" * len(luoi) + "@{}", tieu_de, than,
                       f"TN1 -- tỉ lệ đạt J tốt nhất (\\%) theo số hạt và số vòng trên {tex(_ten_bo(bo, c))}",
                       f"tab:e5-luoi-{bo}",
                       "30 seed mỗi ô; T: số vòng PSO trước khi FCM chạy tới hội tụ.")


def bang_lech(d: pd.DataFrame, bo: str, c: int) -> str:
    luoi = [(n, t) for n in SO_HAT for t in SO_VONG]
    db = d[(d["bo"] == bo) & (d["c"] == c)]
    than = []
    for pb in PHIEN_BAN:
        o = [o_tv_iqr(db.loc[db["phuong_phap"] == _ten_pp(pb, n, t), "khoang_cach_pso"], 1) for n, t in luoi]
        than.append(f"{tex(TEN_PB[pb])} & " + " & ".join(o) + " \\\\")
    diem = db.loc[db["phuong_phap"] == "fcm_diem", "khoang_cach_pso"].median()
    than.append(f"\\multicolumn{{{1 + len(luoi)}}}{{@{{}}l}}{{\\textit{{Đối chứng {tex(DOI_CHUNG['fcm_diem'])}, chưa "
                f"chạy FCM: {so_viet(diem, 1)}}}}} \\\\")
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{{len(SO_VONG)}}}{{c}}{{{n} hạt}}" for n in SO_HAT) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + len(SO_VONG) * i}-{1 + len(SO_VONG) * (i + 1)}}}"
                        for i in range(len(SO_HAT))),
               "Phiên bản & " + " & ".join(f"T = {t}" for _, t in luoi) + " \\\\"]
    return _bang_float("@{}l" + "r" * len(luoi) + "@{}", tieu_de, than,
                       f"TN1 -- J\\textsubscript{{m}} của gbest sau PSO lệch J\\textsubscript{{m}} tốt nhất (\\%) "
                       f"trên {tex(_ten_bo(bo, c))}",
                       f"tab:e5-lech-{bo}",
                       "trung vị và [Q1; Q3] qua 30 seed mỗi ô; đo trước khi FCM chạy; càng nhỏ thì PSO tối ưu càng tốt.", tabcolsep=2)


def kiem_dinh_lech(d: pd.DataFrame) -> pd.DataFrame:
    dong = []
    for (b, c), g in d[d["phien_ban"].isin(PHIEN_BAN)].groupby(["bo", "c"]):
        for n in SO_HAT:
            for t in SO_VONG:
                goc = g[g["phuong_phap"] == _ten_pp("goc", n, t)].set_index("seed")["khoang_cach_pso"]
                for pb in PHIEN_BAN[1:]:
                    x = g[g["phuong_phap"] == _ten_pp(pb, n, t)].set_index("seed")["khoang_cach_pso"]
                    seed = goc.index.intersection(x.index)
                    p = wilcoxon_cap(x[seed], goc[seed])
                    dong.append(dict(bo=b, c=int(c), n=n, t=t, pb=pb, chenh=float((x[seed] - goc[seed]).median()),
                                     p=1.0 if p == "trùng" else p))
    kd = pd.DataFrame(dong)
    kd["p_holm"] = holm(kd["p"])
    kd["tot_hon"] = (kd["p_holm"] < 0.05) & (kd["chenh"] < 0)
    kd["kem_hon"] = (kd["p_holm"] < 0.05) & (kd["chenh"] > 0)
    return kd


def bang_chi_so(d: pd.DataFrame) -> str:
    """Mỗi cấu hình một dòng: chỉ số của FCM k-means++, chênh lệch lớn nhất theo seed của mọi phương pháp cùng nghiệm so
    với FCM k-means++, và khoảng trung vị XB giữa các phương pháp (20 hạt, 30 vòng với PSO)."""
    tk = thong_ke_cung_nghiem(d).set_index(["bo", "c"])
    hien = list(DOI_CHUNG) + [_ten_pp(pb) for pb in PHIEN_BAN]
    than = []
    for b, c in _cau_hinh(d):
        db = d[(d["bo"] == b) & (d["c"] == c)]
        kpp = db[db["phuong_phap"] == "fcm_kpp"]
        xb = db[db["phuong_phap"].isin(hien)].groupby("phuong_phap")["XB"].median()
        r = tk.loc[(b, c)]
        ngan = {"viet_nam": "Việt Nam", "taiwan_kiem_chung": "Taiwan", "taiwan_kiem_chung_khong_log": "Taiwan chưa log"}
        than.append(" & ".join([f"{ngan.get(b, b)}, c = {c}", so_viet(kpp["PC"].median(), 3),
                                so_viet(kpp["silhouette"].median(), 3), so_viet(kpp["auc_rui_ro"].median(), 3),
                                so_gon(r.pc_max, 1), so_gon(r.auc_max, 1), so_viet(r.sil_max, 3),
                                f"{so_gon(xb.min(), 2)} -- {so_gon(xb.max(), 2)}"]) + " \\\\")
    tieu_de = ["& \\multicolumn{3}{c}{FCM (k-means++)} & \\multicolumn{3}{c}{Chênh lớn nhất theo seed} & \\\\",
               "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
               "Cấu hình & PC & Silhouette & AUC & PC & AUC & Silhouette & XB (khoảng) \\\\"]
    return _bang_float("@{}lrrrrrrr@{}", tieu_de, than,
                       "TN1 -- chỉ số sau khi FCM hội tụ: FCM (k-means++) và mức chênh của các phương pháp còn lại",
                       "tab:e5-chi-so",
                       "PC, Silhouette, AUC: trung vị 30 seed của FCM (k-means++); chênh lớn nhất: |hiệu| lớn nhất theo "
                       "từng seed giữa một phương pháp bất kỳ (3 đối chứng, 45 tổ hợp PSO) và FCM (k-means++), chỉ trên các "
                       "lượt cùng nghiệm; XB: khoảng trung vị của 8 phương pháp (3 đối chứng, 5 phiên bản PSO với 20 hạt, "
                       "30 vòng); XB không ổn định khi 2 tâm gần trùng, xem mục~\\ref{sec:e5-chi-so}.")


def thong_ke_cung_nghiem(d: pd.DataFrame) -> pd.DataFrame:
    """Theo (bộ, c, seed): so mọi phương pháp với FCM k-means++ cùng seed — cùng J (≤ TOL_J), chênh PC, AUC, XB."""
    ra = []
    for (b, c), g in d.groupby(["bo", "c"]):
        kpp = g[g["phuong_phap"] == "fcm_kpp"].set_index("seed")
        x = g[g["phuong_phap"] != "fcm_kpp"].join(kpp[["J_m", "PC", "auc_rui_ro", "XB", "silhouette"]], on="seed",
                                                   rsuffix="_kpp")
        cung = (x["J_m"] - x["J_m_kpp"]).abs() <= TOL_J * x["J_m_kpp"]
        xc = x[cung]
        xb = (xc["XB"] / xc["XB_kpp"]).replace([np.inf, -np.inf], np.nan)
        ra.append(dict(bo=b, c=int(c), cung_j=100 * cung.mean(), pc_max=(xc["PC"] - xc["PC_kpp"]).abs().max(),
                       auc_max=(xc["auc_rui_ro"] - xc["auc_rui_ro_kpp"]).abs().max(),
                       sil_max=(xc["silhouette"] - xc["silhouette_kpp"]).abs().max(),
                       xb_ti_le_min=xb.min(), xb_ti_le_max=xb.max(), xb_vo_cung=int(np.isinf(xc["XB"]).sum())))
    return pd.DataFrame(ra)


def hinh_khoang_cach(d: pd.DataFrame, duong_dan: str) -> None:
    plt = _plt()
    ch = _cau_hinh(d)
    fig, axs = plt.subplots(1, len(ch), figsize=(6.3, 2.6), sharey=False)
    for ax, (b, c) in zip(np.atleast_1d(axs), ch):
        db = d[(d["bo"] == b) & (d["c"] == c) & (d["n_hat"] == N_MAC_DINH)]
        for i, pb in enumerate(PHIEN_BAN):
            tv = db[db["phien_ban"] == pb].groupby("so_vong")["khoang_cach_pso"].median()
            ax.plot(tv.index, tv.values, marker="o", ms=3, lw=1, label=TEN_PB[pb], color=plt.get_cmap("tab10")(i))
        ax.set_title(_ten_bo(b, c), fontsize=8)
        ax.set_xlabel("Số vòng PSO")
        ax.set_xticks(list(SO_VONG))
        ax.grid(alpha=0.3)
        _dau_phay(ax, "y")
    np.atleast_1d(axs)[0].set_ylabel("J sau PSO lệch J tốt nhất (%)")
    h, n = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, n, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(duong_dan)
    plt.close(fig)


def macro(d: pd.DataFrame, tt: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuENamSoLuot", so_viet(len(d), 0))]
    for _, r in tt.iterrows():
        pp = r.phuong_phap
        if pp.startswith("pso_"):
            _, pb, n, t = pp.split("_")
            ten_pp = [pb, "Hat", n[1:], "Vong", t[1:]]
        else:
            ten_pp = [pp]
        goc = ["ENam", r.bo, "C", int(r.c), *ten_pp]
        ra += [("soLieu" + ten_macro(*goc, "DatJ"), so_viet(r.dat_j, 0)), ("soLieu" + ten_macro(*goc, "ThoiGian"), so_gon(r.t_tong, 2)),
               ("soLieu" + ten_macro(*goc, "VongFcm"), so_viet(r.vong_fcm, 0)), ("soLieu" + ten_macro(*goc, "Auc"), so_viet(r.auc, 3))]
        if pp != "fcm" and pp != "fcm_kpp":
            ra.append(("soLieu" + ten_macro(*goc, "LechJ"), so_viet(r.khoang_cach_pso, 1)))
    kd = kiem_dinh_lech(d)
    ra += [("soLieuENamKdSoPhep", str(len(kd))), ("soLieuENamKdSoOMoiPb", str(len(kd) // (len(PHIEN_BAN) - 1))),
           ("soLieuENamKdChenhMin", so_viet(kd["chenh"].min(), 1)), ("soLieuENamKdChenhMax", so_viet(kd["chenh"].max(), 1)),
           ("soLieuENamKdPHolmMax", so_viet(kd["p_holm"].max(), 3))]
    for pb, g in kd.groupby("pb"):
        ra += [("soLieu" + ten_macro("ENam", "Kd", pb, "TotHon"), str(int(g["tot_hon"].sum()))),
               ("soLieu" + ten_macro("ENam", "Kd", pb, "KemHon"), str(int(g["kem_hon"].sum())))]
    for _, r in thong_ke_cung_nghiem(d).iterrows():
        goc = ["ENam", r.bo, "C", int(r.c)]
        ra += [("soLieu" + ten_macro(*goc, "CungJKpp"), so_viet(r.cung_j, 1)),
               ("soLieu" + ten_macro(*goc, "PcChenhMax"), so_gon(r.pc_max, 1)),
               ("soLieu" + ten_macro(*goc, "AucChenhMax"), so_gon(r.auc_max, 1)),
               ("soLieu" + ten_macro(*goc, "SilChenhMax"), so_viet(r.sil_max, 3)),
               ("soLieu" + ten_macro(*goc, "XbTiLeMin"), so_gon(r.xb_ti_le_min, 2)),
               ("soLieu" + ten_macro(*goc, "XbTiLeMax"), so_gon(r.xb_ti_le_max, 0)),
               ("soLieu" + ten_macro(*goc, "XbVoCung"), str(r.xb_vo_cung))]
    for (b, c), g in d.groupby(["bo", "c"]):
        kpp = g[g["phuong_phap"] == "fcm_kpp"].set_index("seed")["t_tong"]
        pso = g[g["phuong_phap"].str.endswith(f"_n{N_MAC_DINH}_t{T_MAC_DINH}")]
        ti_le = pso["t_tong"].to_numpy() / kpp.loc[pso["seed"]].to_numpy()
        tl_luoi = pd.Series({pp: np.median(x["t_tong"].to_numpy() / kpp.loc[x["seed"]].to_numpy())
                             for pp, x in g[g["phuong_phap"].str.startswith("pso_")].groupby("phuong_phap")})
        goc = ["ENam", b, "C", int(c)]
        ra += [("soLieu" + ten_macro(*goc, "TiLeThoiGianTv"), so_viet(np.median(ti_le), 1)),
               ("soLieu" + ten_macro(*goc, "PhanTramPsoNhanhHon"), so_viet(100 * (ti_le < 1).mean(), 0)),
               ("soLieu" + ten_macro(*goc, "SoLuotVongNuaKpp"), str(int((g[g["phuong_phap"].str.startswith("pso_")]
                .apply(lambda r: r["n_vong_fcm"] < 0.5 * g.loc[(g["phuong_phap"] == "fcm_kpp") & (g["seed"] == r["seed"]),
                                                               "n_vong_fcm"].iloc[0], axis=1)).sum()))),
               ("soLieu" + ten_macro(*goc, "SoLuotPso"), str(int(g["phuong_phap"].str.startswith("pso_").sum()))),
               ("soLieu" + ten_macro(*goc, "TiLeThoiGianLuoiMin"), so_viet(tl_luoi.min(), 1)),
               ("soLieu" + ten_macro(*goc, "TiLeThoiGianLuoiMax"), so_viet(tl_luoi.max(), 1))]
    tl = d[(d["bo"] == "taiwan_kiem_chung_khong_log") & (d["c"] == 6)]
    if not tl.empty:
        pso_dat = tl.loc[tl["phuong_phap"].str.startswith("pso_"), "dat_j"]
        kpp_dat = tl.loc[tl["phuong_phap"] == "fcm_kpp", "dat_j"]
        ra += [("soLieuENamKhongLogPsoGopDatJ", so_viet(100 * pso_dat.mean(), 0)),
               ("soLieuENamKhongLogAucDat", so_viet(tl.loc[tl["dat_j"], "auc_rui_ro"].median(), 3)),
               ("soLieuENamKhongLogAucKhongDat", so_viet(tl.loc[~tl["dat_j"], "auc_rui_ro"].median(), 3)),
               ("soLieuENamKhongLogSoLuotKhongDat", str(int((~tl["dat_j"]).sum())))]
        tls = tl.set_index("seed")
        kpp_s = tls.loc[tls["phuong_phap"] == "fcm_kpp", "dat_j"]
        mc = [(pp, *mcnemar_cap(g["dat_j"], kpp_s)) for pp, g in tls[tls["phuong_phap"].str.startswith("pso_")]
              .groupby("phuong_phap")]
        ph = holm([x[3] for x in mc])
        ra += [("soLieuENamKhongLogMcSoPhep", str(len(mc))),
               ("soLieuENamKhongLogMcCoYNghia", str(int((ph < 0.05).sum()))),
               ("soLieuENamKhongLogMcPMin", so_viet(min(x[3] for x in mc), 3)),
               ("soLieuENamKhongLogMcPHolmMin", so_viet(ph.min(), 2))]
        ra += macro_doi_chung(["ENam", "KhongLog"], tls)
        for pp in ("fcm", "fcm_diem", "fcm_kpp", _ten_pp("qpso"), _ten_pp("w")):
            x = tl.loc[tl["phuong_phap"] == pp, "dat_j"]
            lo, hi = wilson(int(x.sum()), len(x))
            ten = ["ENam", "KhongLog", pp.replace(f"_n{N_MAC_DINH}_t{T_MAC_DINH}", "")]
            ra += [("soLieu" + ten_macro(*ten, "K"), str(int(x.sum()))),
                   ("soLieu" + ten_macro(*ten, "WilsonDuoi"), so_viet(lo, 1)),
                   ("soLieu" + ten_macro(*ten, "WilsonTren"), so_viet(hi, 1))]
    for (b, c), g in tt[tt["phuong_phap"].str.startswith("pso_")].groupby(["bo", "c"]):
        ra += [("soLieu" + ten_macro("ENam", b, "C", int(c), "Pso", "DatJ", "Min"), so_viet(g["dat_j"].min(), 0)),
               ("soLieu" + ten_macro("ENam", b, "C", int(c), "Pso", "DatJ", "Max"), so_viet(g["dat_j"].max(), 0))]
    return ra


def chay(thu_muc: str, hinh: str) -> list[str]:
    d = doc(thu_muc)
    tt = tom_tat(d)
    bang = os.path.join(thu_muc, "bang")
    os.makedirs(bang, exist_ok=True)
    os.makedirs(hinh, exist_ok=True)
    ra = []
    for ten, noi_dung in [("e5-tong-hop.tex", bang_tong_hop(tt)), ("e5-chi-so.tex", bang_chi_so(d))] + [
            (f"e5-luoi-{b}.tex", bang_luoi(tt, b, c)) for b, c in _cau_hinh(tt)] + [
            (f"e5-lech-{b}.tex", bang_lech(d, b, c)) for b, c in _cau_hinh(tt)]:
        with open(os.path.join(bang, ten), "w", encoding="utf-8") as f:
            f.write(noi_dung)
        ra.append(os.path.join(bang, ten))
    hinh_khoang_cach(d, os.path.join(hinh, "e5-khoang-cach-pso.pdf"))
    ra.append(os.path.join(hinh, "e5-khoang-cach-pso.pdf"))
    ghi_so_lieu(macro(d, tt), os.path.join(thu_muc, "so-lieu-e5.tex"))
    ra.append(os.path.join(thu_muc, "so-lieu-e5.tex"))
    return ra


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ket-qua", default=os.path.join(GOC_TL, "results"))
    ap.add_argument("--hinh", default=os.path.join(GOC_TL, "hinh"))
    a = ap.parse_args(argv)
    for t in chay(os.path.abspath(a.ket_qua), os.path.abspath(a.hinh)):
        print(t)


if __name__ == "__main__":
    main()
