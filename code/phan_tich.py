# -*- coding: utf-8 -*-
"""Phân tích kết quả thực nghiệm: bảng LaTeX, hình, macro số liệu cho Chương 3.

    python3 phan_tich.py [--ket-qua DIR] [--hinh DIR]

Đọc <ket-qua>/E1..E4 và hoi-tu/, ghi <ket-qua>/bang/*.tex, <ket-qua>/so-lieu.tex (\\newcommand cho mọi con số trích
trong văn bản) và <hinh>/*.pdf. Tiêu chí (mục 3.2 của tiểu luận): J_m chỉ so trong loại I; kiểm định Wilcoxon
theo cặp seed so với fcm_kpp, "trùng" khi mọi hiệu bằng 0, Holm trên họ AUC của E1; chỉ số khác báo trung vị [Q1; Q3].
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GOC_TL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KET_QUA_MAC_DINH = os.path.join(GOC_TL, "results")
HINH_MAC_DINH = os.path.join(GOC_TL, "hinh")
TOL_J = 1e-4
ALPHA = 0.05

TEN_HIEN_THI = {
    "kmeans": "K-means",
    "fcm": "FCM",
    "fcm_kpp": "FCM (k-means++)",
    "pso_trong_so": "PSO trọng số",
    "fcm_lap_lai": "FCM × 20",
    "a_L": "A (ω L)",
    "a_S": "A (ω S)",
    "a_LOG": "A (ω LOG)",
    "b_qpso_fcm": "B: QPSO–FCM",
    "cach2_fcm_gieo": "Cách 2: FCM gieo",
    "cach3_xen_ke": "Cách 3: xen kẽ",
    "cach5_pso_U": "Cách 5: PSO trên U",
    "de_xuat": "Đề xuất",
    "de_xuat_ep_pso": "Đề xuất (ép PSO)",
    "c_msgwo_wfcm": "C: MSGWO–WFCM",
    "wfcm_giai_tich": "WFCM giải tích",
    "cach7_pso_trong_so": "Cách 7: PSO trọng số",
    "kfcm": "KFCM",
    "cach8_pso_kfcm": "Cách 8: PSO–KFCM",
}
TEN_BO = {
    "viet_nam": "Việt Nam",
    "taiwan_kiem_chung": "Taiwan Default",
    "taiwan_kiem_chung_khong_log": "Taiwan Default (không log)",
    "south_german": "South German Credit",
    "german_so": "German Credit (bản số)",
    "australian": "Australian Credit",
    "polish": "Polish Bankruptcy",
    "taiwan_pha_san": "Taiwan Bankruptcy",
}
NHOM = [
    ("Đối chứng", ["kmeans", "fcm", "fcm_kpp", "fcm_lap_lai"]),
    ("Loại I: PSO tối ưu J\\textsubscript{m}", ["a_L", "a_S", "a_LOG", "b_qpso_fcm", "cach2_fcm_gieo", "cach3_xen_ke", "cach5_pso_U",
                                "de_xuat", "de_xuat_ep_pso"]),
    ("Loại II: tối ưu hàm khác", ["c_msgwo_wfcm", "wfcm_giai_tich", "cach7_pso_trong_so", "kfcm", "cach8_pso_kfcm"]),
]
# Bảng và hình trong bài chỉ hiện các phương pháp được trình bày; thống kê, Holm và macro vẫn tính trên đủ NHOM
# (họ phép so đã chốt trước khi chạy), nên số liệu không đổi khi thu gọn phần trình bày.
NHOM_HIEN_THI = [
    ("Đối chứng", ["kmeans", "fcm", "fcm_kpp"]),
    ("Loại I: PSO tối ưu J\\textsubscript{m}", ["a_L", "a_S", "a_LOG", "b_qpso_fcm", "de_xuat", "de_xuat_ep_pso"]),
    ("Tối ưu hàm khác", ["c_msgwo_wfcm"]),
]
HIEN_THI = {p for _, nhom in NHOM_HIEN_THI for p in nhom}
CO_J = {"fcm", "fcm_kpp", "fcm_lap_lai"} | set(NHOM[1][1])  # J_m chỉ so trong cùng hàm mục tiêu
CO_LICH_SU = ["a_L", "a_S", "a_LOG", "b_qpso_fcm", "cach2_fcm_gieo", "cach3_xen_ke", "cach5_pso_U", "de_xuat"]
SO_TU = {0: "Khong", 1: "Mot", 2: "Hai", 3: "Ba", 4: "Bon", 5: "Nam", 6: "Sau", 7: "Bay", 8: "Tam", 9: "Chin"}
GHI_CHU_E1 = ("AUC tách rủi ro: cụm rủi ro là cụm cứng có tỉ lệ vỡ nợ cao nhất (chọn bằng nhãn, như nhau cho mọi "
              "phương pháp), điểm rủi ro là độ thuộc vào cụm đó; khi các tâm gần trùng, độ thuộc không tăng theo rủi "
              "ro nên AUC có thể dưới 0,5. Với K-means (độ thuộc one-hot) PC = 1, PE = 0 là tầm thường. "
              "J\\textsubscript{m} chỉ so giữa các phương pháp cùng hàm mục tiêu; C tối ưu hàm khác nên để trống (--).")


# ------------------------------------------------------------------ thống kê
def holm(p) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni: p_(i)·(m − i + 1) theo thứ tự tăng, lấy cực đại tích luỹ, chặn 1."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    thu_tu = np.argsort(p)
    hc = np.minimum(1.0, np.maximum.accumulate(p[thu_tu] * (m - np.arange(m))))
    kq = np.empty(m)
    kq[thu_tu] = hc
    return kq


def wilcoxon_cap(a, b):
    """Wilcoxon theo cặp; trả "trùng" khi mọi hiệu bằng 0 (không tính p)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(d == 0):
        return "trùng"
    with warnings.catch_warnings():  # ít hiệu khác 0: scipy cảnh báo xấp xỉ chuẩn, p vẫn tính được
        warnings.simplefilter("ignore", UserWarning)
        return float(wilcoxon(a, b).pvalue)


def so_sanh_cap(d: pd.DataFrame, nhom: list[str], chinh: str, doi_chung: list[str], chi_so: list[str],
                alpha: float = 0.05) -> pd.DataFrame:
    """So phương pháp `chinh` với từng đối chứng theo cặp seed: mỗi (nhóm, chỉ số, đối chứng) một dòng gồm trung vị hiệu
    chinh − đối chứng, p Wilcoxon (hiệu toàn 0 thì p = 1), p Holm trên toàn bộ các dòng và kết luận +1 / 0 / −1."""
    hang = []
    for khoa, g in d.groupby(nhom):
        p = g.pivot(index="seed", columns="phuong_phap")
        for cs in chi_so:
            for dc in doi_chung:
                hieu = (p[cs][chinh] - p[cs][dc]).dropna()
                pv = wilcoxon_cap(hieu.to_numpy(), np.zeros(len(hieu)))
                hang.append({**dict(zip(nhom, khoa if isinstance(khoa, tuple) else (khoa,))), "chi_so": cs,
                             "doi_chung": dc, "hieu_tv": float(hieu.median()), "p": 1.0 if isinstance(pv, str) else pv})
    kd = pd.DataFrame(hang)
    kd["p_holm"] = holm(kd["p"].to_numpy())
    kd["ket_luan"] = np.where(kd["p_holm"] < alpha, np.sign(kd["hieu_tv"]), 0).astype(int)
    return kd


DAU_KL = {1: "\\textsuperscript{+}", -1: "\\textsuperscript{--}", 0: ""}


def kiem_dinh_e1(e1: pd.DataFrame, chi_so: str = "auc_rui_ro", goc: str = "fcm_kpp",
                 gom_cung_j: bool = True) -> pd.DataFrame:
    """Mỗi (bộ, c, phương pháp ≠ goc): trung vị hiệu theo cặp seed, p Wilcoxon, p Holm trên cả họ của E1.

    Phương pháp cùng hàm mục tiêu với goc (CO_J): cặp seed có |ΔJ_m| ≤ TOL_J (tương đối) là cùng nghiệm — đúng định
    nghĩa "cùng kết quả với FCM" — nên hiệu chỉ số tính bằng 0. Không làm vậy thì nhiễu hội tụ (FCM dừng ở ε = 1e-5,
    |ΔAUC| ≤ 3e-5) bị Wilcoxon coi là khác biệt. gom_cung_j=False: quy tắc nguyên văn (so 0 tuyệt đối), để báo song song.
    """
    dong = []
    for (bo, c), d in e1.groupby(["bo", "c"]):
        g = d[d["phuong_phap"] == goc].set_index("seed")
        for pp, dp in d[d["phuong_phap"] != goc].groupby("phuong_phap"):
            x = dp.set_index("seed")
            seed = x.index.intersection(g.index)
            hieu = (x.loc[seed, chi_so] - g.loc[seed, chi_so]).to_numpy(copy=True)
            if gom_cung_j and pp in CO_J and goc in CO_J:
                j = g.loc[seed, "J_m"].to_numpy()
                hieu[np.abs(x.loc[seed, "J_m"].to_numpy() - j) <= TOL_J * np.abs(j)] = 0.0
            hieu = hieu[np.isfinite(hieu)]  # cặp có dòng lỗi (NaN) không vào phép so
            p = wilcoxon_cap(hieu, np.zeros_like(hieu))
            dong.append(dict(bo=bo, c=c, phuong_phap=pp, so_cap=len(hieu), hieu_tv=float(np.median(hieu)),
                             p=np.nan if p == "trùng" else p, trung=p == "trùng"))
    kq = pd.DataFrame(dong)
    co_p = ~kq["trung"]
    kq["p_holm"] = np.nan
    kq.loc[co_p, "p_holm"] = holm(kq.loc[co_p, "p"].values)
    return kq


def holm_toan_ho(kd: pd.DataFrame) -> np.ndarray:
    """Holm trên mọi phép so, kể cả phép "trùng" (tính p = 1) — kiểm độ nhạy theo cỡ họ."""
    return holm(np.where(kd["trung"], 1.0, kd["p"]))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Khoảng tin cậy 95% Wilson cho tỉ lệ k/n, tính bằng %; dùng được cả khi k = 0 hoặc k = n."""
    p = k / n
    tam = (p + z * z / (2 * n)) / (1 + z * z / n)
    nua = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, tam - nua), 100 * min(1.0, tam + nua)


def mcnemar_cap(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """McNemar chính xác hai phía cho hai dãy đạt/không đạt ghép theo seed (index là seed).

    Trả (số seed chỉ a đạt, số seed chỉ b đạt, p); p = 1 khi không có cặp bất đồng."""
    from scipy.stats import binomtest

    chung = a.index.intersection(b.index)
    a, b = a.loc[chung].astype(bool), b.loc[chung].astype(bool)
    chi_a, chi_b = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if chi_a + chi_b == 0 else float(binomtest(chi_a, chi_a + chi_b, 0.5).pvalue)
    return chi_a, chi_b, p


def so_voi_doi_chung(dc: pd.DataFrame) -> dict[str, float]:
    """So mọi tổ hợp PSO với 3 đối chứng FCM trên tỉ lệ đạt J tốt nhất, ghép theo seed.

    dc: các lượt của một cấu hình, index là seed, có cột phuong_phap và dat_j. Với FCM U ngẫu nhiên và FCM c điểm ngẫu
    nhiên: McNemar từng tổ hợp, Holm trên các tổ hợp, đếm tổ hợp tốt hơn/kém hơn có ý nghĩa; với c điểm ngẫu nhiên còn
    gộp số lượt PSO đạt khi đối chứng trượt (thắng) và ngược lại (thua). Với k-means++: số seed k-means++ trượt, cận trên
    Wilson của tỉ lệ trượt (mức lợi ích lớn nhất một bộ khởi tạo khác có thể có) và hiệu tỉ lệ đạt PSO − k-means++ theo
    tổ hợp (điểm %)."""
    pso = dc[dc["phuong_phap"].str.startswith("pso_")]
    ra: dict[str, float] = {"SoToHop": float(pso["phuong_phap"].nunique()), "SoLuotPso": float(len(pso))}
    for dc_ten, ten in (("fcm", "U"), ("fcm_diem", "Diem")):
        dch = dc.loc[dc["phuong_phap"] == dc_ten, "dat_j"]
        mc = [mcnemar_cap(g["dat_j"], dch) for _, g in pso.groupby("phuong_phap")]
        ph = holm([x[2] for x in mc])
        ra[ten + "TotHon"] = float(sum(1 for x, q in zip(mc, ph) if q < 0.05 and x[0] > x[1]))
        ra[ten + "KemHon"] = float(sum(1 for x, q in zip(mc, ph) if q < 0.05 and x[0] < x[1]))
        ra[ten + "Thang"] = float(sum(x[0] for x in mc))
        ra[ten + "Thua"] = float(sum(x[1] for x in mc))
    kpp = dc.loc[dc["phuong_phap"] == "fcm_kpp", "dat_j"].astype(bool)
    truot = int((~kpp).sum())
    ra["KppTruot"] = float(truot)
    ra["KppTruotWilsonTren"] = wilson(truot, len(kpp))[1]
    hieu = 100 * (pso.groupby("phuong_phap")["dat_j"].mean() - kpp.mean())
    ra["HieuKppMin"], ra["HieuKppMax"] = float(hieu.min()), float(hieu.max())
    return ra


def macro_doi_chung(goc: list, dc: pd.DataFrame) -> list[tuple[str, str]]:
    """Macro \\soLieu<goc>DoiChung<ten> từ so_voi_doi_chung; số đếm in nguyên, cận Wilson và hiệu in một chữ số thập phân."""
    kq = so_voi_doi_chung(dc)
    return [("soLieu" + ten_macro(*goc, "DoiChung", k), so_viet(v, 1 if k in ("KppTruotWilsonTren", "HieuKppMin",
                                                                                "HieuKppMax") else 0))
            for k, v in kq.items()]


def so_cuc_tieu(J, tol: float = TOL_J) -> int:
    """Số mức J khác nhau: sắp tăng, mở mức mới khi J vượt mức hiện tại quá tol (tương đối)."""
    J = np.sort(np.asarray(J, dtype=float))
    muc, dau = 0, None
    for j in J:
        if dau is None or j > dau * (1 + tol):
            muc, dau = muc + 1, j
    return muc


def ti_le_dat_j_tot_nhat(df: pd.DataFrame, nhom=("bo", "c"), theo="phuong_phap") -> pd.Series:
    j_min = df.groupby(list(nhom))["J_m"].transform("min")
    return (df["J_m"] <= (1 + TOL_J) * j_min).groupby([df[k] for k in (*nhom, theo)]).mean()


# ------------------------------------------------------------------ định dạng
def so_viet(x, so_le: int = 2) -> str:
    """Dấu chấm phân nghìn, dấu phẩy thập phân, dấu trừ Unicode; NaN → "--"."""
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "--"
    s = f"{abs(float(x)):,.{so_le}f}".replace(",", "#").replace(".", ",").replace("#", ".")
    return ("−" if float(x) < 0 and s.strip("0,.") else "") + s


def so_gon(x, so_le: int = 3, nguong: float = 1e6) -> str:
    """Như so_viet nhưng chuyển sang dạng a·10^k khi |x| rất lớn hoặc quá nhỏ để hiện với so_le chữ số."""
    if x is not None and np.isinf(x):
        return "∞" if x > 0 else "−∞"
    if x is None or not np.isfinite(x):
        return "--"
    if x != 0 and (abs(x) >= nguong or abs(x) < 0.5 * 10 ** (-so_le)):
        k = int(np.floor(np.log10(abs(x))))
        return f"{so_viet(x / 10**k, 2)}·10\\textsuperscript{{{'−' if k < 0 else ''}{abs(k)}}}"
    return so_viet(x, so_le)


def phan_vi(s, q=(25, 50, 75)) -> np.ndarray:
    """Phân vị bỏ NaN, dùng chung cho bảng và macro. ∞ (XB khi 2 tâm trùng) được thay bằng số rất lớn để nội suy
    tuyến tính không sinh NaN (∞ − ∞), rồi trả lại ∞ — phân vị rơi vào đoạn có ∞ là ∞."""
    x = np.asarray(pd.Series(s).dropna(), dtype=float)
    if x.size == 0:
        return np.full(len(q), np.nan)
    kq = np.percentile(np.where(np.isposinf(x), 1e306, x), q)
    return np.where(kq > 1e300, np.inf, kq)


def tex(s: str) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def o_tv_iqr(s: pd.Series, so_le: int = 3, he_so: float = 1.0, nguong: float = 1e6) -> str:
    """Ô 2 dòng: trung vị, dưới là [Q1; Q3] cỡ chữ nhỏ."""
    s = s.dropna() * he_so
    if s.empty:
        return "--"
    q1, tv, q3 = phan_vi(s)
    return (f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{so_gon(tv, so_le, nguong)}\\\\"
            f"{{\\scriptsize[{so_gon(q1, so_le, nguong)}; {so_gon(q3, so_le, nguong)}]}}\\end{{tabular}}")


def hai_dong(tren: str, duoi: str) -> str:
    """Tiêu đề cột 2 dòng (không cần gói makecell)."""
    return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{tren}\\\\{duoi}\\end{{tabular}}"


def ten_macro(*phan) -> str:
    """Ghép các phần thành tên chỉ gồm chữ cái ASCII: tách theo ký tự không phải chữ/số, viết hoa đầu, số thành chữ."""
    kq = []
    for p in phan:
        for tu in re.split(r"[^A-Za-z0-9]+", str(p)):
            for doan in re.findall(r"[A-Za-z]+|[0-9]+", tu):
                kq.append("".join(SO_TU[int(ch)] for ch in doan) if doan.isdigit() else doan[0].upper() + doan[1:])
    return "".join(kq)


def _bang_float(cot: str, tieu_de: list[str], than: list[str], caption: str, nhan: str, ghi_chu: str = "",
                tabcolsep: float = 3) -> str:
    return "\n".join([
        "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{nhan}}}", "\\footnotesize",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep:g}pt}}", f"\\begin{{tabular}}{{{cot}}}", "\\toprule", *tieu_de, "\\midrule", *than,
        "\\bottomrule", "\\end{tabular}",
        *([f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize Ghi chú: {ghi_chu}}}"] if ghi_chu else []),
        "\\end{table}", ""])


def _doc(thu_muc: str, e: str) -> pd.DataFrame:
    tep = sorted(glob.glob(os.path.join(thu_muc, e, "*.csv")))
    return pd.concat([pd.read_csv(t) for t in tep], ignore_index=True) if tep else pd.DataFrame()


def _tep_mo_ta(thu_muc: str) -> str | None:
    return next((t for t in (os.path.join(thu_muc, "mo-ta-du-lieu.csv"),
                             os.path.join(KET_QUA_MAC_DINH, "mo-ta-du-lieu.csv")) if os.path.exists(t)), None)


def _co_mat(df: pd.DataFrame, nhom) -> list[str]:
    co = set(df["phuong_phap"])
    return [p for p in nhom if p in co]


# ------------------------------------------------------------------ bảng
def bang_mo_ta(mo_ta: pd.DataFrame) -> str:
    than = [f"{tex(TEN_BO.get(r.bo, r.bo))} & {so_viet(r.n_dung, 0)} & {r.D_dung} & {so_viet(r['nhan1_%'], 1)} & "
            f"{so_viet(r.mat_can_bang, 1)} & {so_viet(r.lech_tv, 2)} & {so_viet(r.nhon_tv, 1)} & "
            f"{so_viet(r['ngoai_lai_%'], 1)} & {r['cap_r>0,9']} & {r['D_hieu_dung_90%']} \\\\"
            for _, r in mo_ta.iterrows()]
    tieu_de = ["Bộ dữ liệu & $n$ & $D$ & " + " & ".join([
        hai_dong("Nhãn 1", "(\\%)"), hai_dong("Mất", "cân bằng"), hai_dong("Bất", "đối xứng"), hai_dong("Nhọn", "dư"),
        hai_dong("Ngoại lai", "(\\%)"), hai_dong("Cặp", "$|r|>0{,}9$"), "$D_{90\\%}$"]) + " \\\\"]
    return _bang_float("@{}lrrrrrrrrr@{}", tieu_de, than, "Hồ sơ các bộ dữ liệu sau tiền xử lý", "tab:mo-ta-du-lieu",
                       "mất cân bằng: số quan sát nhãn 0 trên số quan sát nhãn 1; bất đối xứng, nhọn dư: trung vị $|$hệ số bất đối "
                       "xứng$|$ và độ nhọn dư các thuộc tính sau tiền xử lý; ngoại lai: "
                       "tỉ lệ điểm có ít nhất một thuộc tính $|z|>3$; $D_{90\\%}$: số thành phần chính giữ 90\\% "
                       "phương sai.")


def bang_e1_ket_qua(e1: pd.DataFrame, bo: str) -> str:
    d = e1[e1["bo"] == bo]
    cot = [("J_m", 1, 1.0, 1e6), ("auc_rui_ro", 3, 1.0, 1e6), ("silhouette", 3, 1.0, 1e6),
           ("cum_nho_nhat", 1, 100.0, 1e6), ("PC", 3, 1.0, 1e6), ("PE", 3, 1.0, 1e6), ("XB", 2, 1.0, 1e4)]
    than = []
    for c, dc in d.groupby("c"):
        than.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{c = {c}}}}} \\\\")
        for ten_nhom, nhom in NHOM_HIEN_THI:
            co = _co_mat(dc, nhom)
            if not co:
                continue
            than.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{ten_nhom}}}}} \\\\")
            for pp in co:
                dp = dc[dc["phuong_phap"] == pp]
                o = [o_tv_iqr(dp[k], sl, hs, ng) if (k != "J_m" or pp in CO_J) and k in dp else "--"
                     for k, sl, hs, ng in cot]
                than.append(f"{TEN_HIEN_THI[pp]} & " + " & ".join(o) + " \\\\")
        than.append("\\midrule")
    than = than[:-1]
    tieu_de = ("Phương pháp & J\\textsubscript{m} & AUC & Silhouette & "
               + hai_dong("Cụm nhỏ", "nhất (\\%)") + " & PC & PE & XB \\\\")
    return "\n".join([
        "{\\scriptsize", "\\setlength{\\tabcolsep}{1.2pt}", "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.3cm}ccccccc@{}}",
        f"\\caption{{E1 -- chất lượng phân cụm trên {tex(TEN_BO.get(bo, bo))}: trung vị [Q1; Q3] qua 30 seed}}",
        f"\\label{{tab:e1-ket-qua-{bo}}}\\\\", "\\toprule", tieu_de, "\\midrule", "\\endfirsthead",
        "\\toprule", tieu_de, "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot",
        *than, "\\end{longtable}", "}", f"{{\\scriptsize\\noindent Ghi chú: {GHI_CHU_E1}\\par}}", ""])


def bang_e1_thoi_gian(e1: pd.DataFrame, bo: str) -> str:
    d = e1[e1["bo"] == bo]
    cot = [("t_khoi_tao", 3), ("t_tim_kiem", 3), ("t_tinh_chinh", 3), ("t_tong", 3), ("n_fe", 0), ("n_vong_fcm", 0)]
    than = []
    for c, dc in d.groupby("c"):
        than.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{c = {c}}}}} \\\\")
        for _, nhom in NHOM_HIEN_THI:
            for pp in _co_mat(dc, nhom):
                dp = dc[dc["phuong_phap"] == pp]
                than.append(f"{TEN_HIEN_THI[pp]} & " + " & ".join(so_gon(dp[k].median(), sl) for k, sl in cot) + " \\\\")
        than.append("\\midrule")
    than = than[:-1]
    tieu_de = "Phương pháp & Khởi tạo (s) & Tìm kiếm (s) & Tinh chỉnh (s) & Tổng (s) & $n_{FE}$ & Số vòng FCM \\\\"
    return "\n".join([
        "{\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{longtable}{@{}lrrrrrr@{}}",
        f"\\caption{{E1 -- thời gian theo pha và chi phí trên {tex(TEN_BO.get(bo, bo))}: trung vị qua 30 seed}}",
        f"\\label{{tab:e1-thoi-gian-{bo}}}\\\\", "\\toprule", tieu_de, "\\midrule", "\\endfirsthead",
        "\\toprule", tieu_de, "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot",
        *than, "\\end{longtable}", "}",
        "{\\scriptsize\\noindent Ghi chú: thời gian thực là thước đo chi phí chính; $n_{FE}$ không đồng đơn vị giữa "
        "các phương pháp (Mục~\\ref{sec:thiet-lap}), chỉ để tham khảo.\\par}", ""])


def doi_chieu_kiem_dinh(e1: pd.DataFrame, kd: pd.DataFrame) -> dict:
    """Số liệu kiểm độ nhạy của quy tắc "trùng": quy tắc nguyên văn (so 0 tuyệt đối) và Holm trên toàn họ."""
    cu = kiem_dinh_e1(e1, gom_cung_j=False)
    toan_ho = holm_toan_ho(kd)
    doi = kd[(kd["p_holm"] < ALPHA) & (toan_ho >= ALPHA)]
    return dict(cu_trung=int(cu["trung"].sum()), cu_co_p=int((~cu["trung"]).sum()),
                cu_khac=int((cu["p_holm"] < ALPHA).sum()), toan_ho_khac=int((toan_ho < ALPHA).sum()),
                o_bien=[f"{TEN_HIEN_THI[r.phuong_phap]} {TEN_BO.get(r.bo, r.bo)} c = {r.c}" for r in doi.itertuples()])


def bang_e1_kiem_dinh(kd: pd.DataFrame, dc: dict | None = None) -> str:
    cau_hinh = sorted({(b, c) for b, c in zip(kd["bo"], kd["c"])})
    than = []
    for _, nhom in NHOM_HIEN_THI:
        for pp in [p for p in nhom if p in set(kd["phuong_phap"])]:
            o = []
            for b, c in cau_hinh:
                r = kd[(kd["bo"] == b) & (kd["c"] == c) & (kd["phuong_phap"] == pp)]
                if r.empty:
                    o.append("--")
                    continue
                r = r.iloc[0]
                if r.trung:
                    o.append("trùng")
                else:
                    sao = "$^{*}$" if r.p_holm < ALPHA else ""
                    o.append(f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{so_viet(r.hieu_tv, 3)}{sao}\\\\"
                             f"{{\\tiny p = {so_gon(r.p_holm, 3)}}}\\end{{tabular}}")
            than.append(f"{TEN_HIEN_THI[pp]} & " + " & ".join(o) + " \\\\")
    n = len(cau_hinh)
    bo_ds = list(dict.fromkeys(b for b, _ in cau_hinh))
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{{sum(1 for b2, _ in cau_hinh if b2 == b)}}}{{c}}"
                                 f"{{{tex(TEN_BO.get(b, b))}}}" for b in bo_ds) + " \\\\",
               "Phương pháp & " + " & ".join(f"c = {c}" for _, c in cau_hinh) + " \\\\"]
    return _bang_float("@{}l" + "c" * n + "@{}", tieu_de, than,
                       "E1 -- AUC tách rủi ro so với FCM (k-means++): trung vị hiệu theo cặp seed và p Holm",
                       "tab:e1-kiem-dinh",
                       "Wilcoxon theo cặp seed; với phương pháp cùng hàm mục tiêu J\\textsubscript{m}, cặp seed có "
                       "J\\textsubscript{m} lệch tương đối $\\le 10^{-4}$ (cùng nghiệm) tính hiệu bằng 0; "
                       "hiệu chỉnh Holm trên toàn họ phép so AUC của E1 đã chốt trước "
                       f"({int((~kd['trung']).sum())} phép so có p, gồm cả các phương pháp phụ không in trong bảng); "
                       "$^{*}$: p Holm $<$ 0,05; "
                       "``trùng'': mọi hiệu bằng 0, không tính p."
                       + ("" if dc is None else
                          f" Quy tắc làm rõ sau khi chạy; theo nguyên văn (so 0 tuyệt đối): {dc['cu_trung']} trùng, "
                          f"{dc['cu_khac']} khác biệt trên {dc['cu_co_p']} phép so có p. Holm trên đủ "
                          f"{len(kd)} phép so (trùng tính p = 1): {dc['toan_ho_khac']} khác biệt"
                          + (f"; đổi kết luận ở {', '.join(dc['o_bien'])}" if dc["o_bien"] else "") + "."))


def bang_e2(e2: pd.DataFrame) -> str:
    tl = ti_le_dat_j_tot_nhat(e2)
    cs = sorted(e2["c"].unique())
    than = []
    for _, nhom in NHOM_HIEN_THI:
        for pp in _co_mat(e2, nhom):
            o = []
            for c in cs:
                dp = e2[(e2["c"] == c) & (e2["phuong_phap"] == pp)]
                bo = dp["bo"].iloc[0]
                o += [so_viet(100 * tl.loc[(bo, c, pp)], 0), so_viet(dp["auc_rui_ro"].median(), 3),
                      so_viet(100 * dp["cum_nho_nhat"].median(), 1), so_gon(dp["t_tong"].median(), 2)]
            than.append(f"{TEN_HIEN_THI[pp]} & " + " & ".join(o) + " \\\\")
    tieu_de = ["& " + " & ".join(f"\\multicolumn{{4}}{{c}}{{c = {c}}}" for c in cs) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(cs))),
               "Phương pháp" + (" & " + " & ".join([hai_dong("Đạt J", "tốt nhất (\\%)"), "AUC",
                                                       hai_dong("Cụm nhỏ", "nhất (\\%)"), hai_dong("Tổng", "(s)")])
                                 ) * len(cs) + " \\\\"]
    bo = TEN_BO.get(e2["bo"].iloc[0], e2["bo"].iloc[0])
    return _bang_float("@{}l" + "rrrr" * len(cs) + "@{}", tieu_de, than,
                       f"E2 -- ít cụm và nhiều cụm trên {tex(bo)}: tỉ lệ đạt J tốt nhất và chi phí (30 seed)",
                       "tab:e2-it-nhieu-cum",
                       "``đạt J tốt nhất'': J\\textsubscript{m} $\\le (1 + 10^{-4})$ lần J\\textsubscript{m} nhỏ nhất "
                       "của mọi lượt mọi phương pháp cùng c (kể cả phương pháp phụ không in); AUC, cụm nhỏ nhất, "
                       "thời gian là trung vị.")


def thong_ke_e3(e3: pd.DataFrame) -> pd.DataFrame:
    dong = []
    for (bo, c), d in e3.groupby(["bo", "c"], sort=False):
        j_min = d["J_m"].min()
        r = dict(bo=bo, c=c)
        for kieu, dk in d.groupby("kieu"):
            r[f"{kieu}_so_cuc_tieu"] = so_cuc_tieu(dk["J_m"])
            r[f"{kieu}_dat_j"] = float((dk["J_m"] <= (1 + TOL_J) * j_min).mean())
            r[f"{kieu}_sup"] = float(dk["sup"].mean())
            r[f"{kieu}_pc"] = float(dk["PC"].median())
        dong.append(r)
    thu_tu = {b: i for i, b in enumerate(TEN_BO)}
    return pd.DataFrame(dong).sort_values(["bo", "c"], key=lambda s: s.map(thu_tu) if s.name == "bo" else s,
                                          ignore_index=True)


def bang_e3(e3: pd.DataFrame) -> str:
    tk = thong_ke_e3(e3)
    than = []
    for _, r in tk.iterrows():
        than.append(f"{tex(TEN_BO.get(r.bo, r.bo))} & {r.c} & " + " & ".join(
            f"{r[f'{k}_so_cuc_tieu']} & {so_viet(100 * r[f'{k}_dat_j'], 0)} & {so_viet(100 * r[f'{k}_sup'], 0)}"
            for k in ("u_ngau_nhien", "kmeans_pp")) + " \\\\")
    tieu_de = ["& & \\multicolumn{3}{c}{U ngẫu nhiên} & \\multicolumn{3}{c}{k-means++} \\\\",
               "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
               "Bộ dữ liệu & c" + (" & " + " & ".join([hai_dong("Số", "cực tiểu"), hai_dong("Đạt J", "tốt nhất (\\%)"),
                                                     hai_dong("Sụp", "(\\%)")])) * 2 + " \\\\"]
    n_kt = int(e3.groupby(["bo", "c", "kieu"]).size().max())
    return _bang_float("@{}lrrrrrrr@{}", tieu_de, than,
                       f"TN2 -- chẩn đoán số cực tiểu của J\\textsubscript{{m}}: FCM từ {n_kt} khởi tạo mỗi kiểu",
                       "tab:e3-chan-doan",
                       "số cực tiểu: số mức J\\textsubscript{m} lệch nhau quá $10^{-4}$ (tương đối), kể cả mức sụp; "
                       "đạt J tốt nhất so với J\\textsubscript{m} nhỏ nhất của cả 2 kiểu; sụp: mọi tâm dồn về "
                       "trung bình dữ liệu ($\\max|V| < 10^{-3}$).")


def chon_thuoc_tinh(e4: pd.DataFrame, bo: str, do_lech: pd.Series, so: int = 5, pp: str = "pso_trong_so") -> list[str]:
    """Thuộc tính có khoảng trung bình cụm (theo phân hoạch pp) rộng nhất, tính theo độ lệch chuẩn của thuộc tính."""
    d = e4[(e4["bo"] == bo) & (e4["phuong_phap"] == pp)]
    pham_vi = (d[do_lech.index].max() - d[do_lech.index].min()) / do_lech.replace(0, np.nan)
    return list(pham_vi.sort_values(ascending=False).index[:so])


Y_NGHIA_VN = {  # data/README.md
    "X1": "tỉ số tiền mặt", "X2": "thanh toán nhanh", "X4": "nợ dài hạn/VCSH", "X5": "nợ dài hạn/tổng tài sản",
    "X6": "tổng nợ/VCSH", "X7": "tổng nợ/tổng tài sản", "X8": "nợ ngắn hạn/VCSH", "X9": "nợ ngắn hạn/tổng tài sản",
    "X10": "phải trả người bán/VCSH", "X11": "phải trả người bán/tổng tài sản", "X12": "tổng tài sản/tổng nợ",
    "X13": "EBITDA/(nợ ngắn hạn + lãi vay)", "X14": "P/E", "X16": "P/B", "X17": "P/S", "X19": "vốn hoá thị trường",
    "X20": "P/CF", "X22": "EV/doanh thu", "X23": "EV/EBITDA", "X24": "EV/EBIT", "X25": "EPS pha loãng",
}
GIAI_NGHIA = {
    "viet_nam": None,  # sinh theo thuộc tính được chọn, xem bang_e4
    "taiwan_kiem_chung": "PAY\\_k: tình trạng trả nợ ở tháng thứ k trước tháng quan sát (theo mô tả của UCI: $-1$ trả "
                         "đúng hạn, $1$ đến $9$ trễ hạn 1 đến 9 tháng; dữ liệu còn các mã $-2$ và $0$ không được giải "
                         "thích); giá trị dương là có trễ hạn.",
}


def bang_e4(e4: pd.DataFrame, bo: str, thuoc_tinh: list[str]) -> str:
    d = e4[e4["bo"] == bo]
    n = d.groupby("phuong_phap")["kich_thuoc"].sum().iloc[0]
    than = []
    for pp in [p for p in ("fcm_kpp", "pso_trong_so") if p in set(d["phuong_phap"])]:
        dp = d[d["phuong_phap"] == pp].sort_values("cum")
        for i, (_, r) in enumerate(dp.iterrows()):
            than.append(f"{TEN_HIEN_THI[pp] if i == 0 else ''} &{int(r.cum) + 1} & {so_viet(100 * r.kich_thuoc / n, 1)} & "
                        f"{so_viet(100 * r.ti_le_nhan1, 1)} & {so_viet(r['mo_ho_%'], 1)} & "
                        + " & ".join(so_gon(r[t], 2) for t in thuoc_tinh) + " \\\\")
        than.append("\\midrule")
    than = than[:-1]
    tieu_de = ["Phương pháp & Cụm & Cỡ (\\%) & Vỡ nợ (\\%) & Mơ hồ (\\%) & "
               + " & ".join(f"\\rotatebox{{60}}{{{tex(t)}}}" for t in thuoc_tinh) + " \\\\"]
    return _bang_float("@{}lrrrr" + "r" * len(thuoc_tinh) + "@{}", tieu_de, than,
                       f"Profile cụm FCM thường và FCM có trọng số do PSO tìm trên {tex(TEN_BO.get(bo, bo))} (c = 3, seed 1000): "
                       "trung bình giá trị gốc",
                       f"tab:e4-profile-{bo}",
                       "cụm đánh số theo tỉ lệ vỡ nợ tăng dần; mơ hồ: tỉ lệ điểm trong cụm có độ thuộc lớn nhất "
                       "$<$ 0,6; thuộc tính chọn là 5 thuộc tính có khoảng trung bình giữa các cụm (PSO trọng số) "
                       "rộng nhất tính theo độ lệch chuẩn. " + (
                           "; ".join(f"{t}: {Y_NGHIA_VN[t]}" for t in thuoc_tinh if t in Y_NGHIA_VN) + "."
                           if bo == "viet_nam" else GIAI_NGHIA.get(bo, "")))


# ------------------------------------------------------------------ macro số liệu
def macro_so_lieu(thu_muc: str) -> list[tuple[str, str]]:
    m: list[tuple[str, str]] = []

    def them(gia_tri: str, *phan):
        m.append(("soLieu" + ten_macro(*phan), gia_tri))

    e1, e2, e3, e4 = (_doc(thu_muc, e) for e in ("E1", "E2", "E3", "E4"))
    mo_ta = _tep_mo_ta(thu_muc)
    if mo_ta:
        for _, r in pd.read_csv(mo_ta).iterrows():
            k = ("MoTa", r.bo)
            them(so_viet(r["nhan1_%"], 1), *k, "NhanMot")
            them(so_viet(r.lech_tv, 2), *k, "Lech")
            them(so_viet(r.nhon_tv, 1), *k, "Nhon")
            them(so_viet(r["ngoai_lai_%"], 1), *k, "NgoaiLai")
            them(str(r["D_hieu_dung_90%"]), *k, "DHieuDung")
            them(str(r.D_dung), *k, "D")
    if len(e1):
        them(so_viet(len(e1), 0), "EMot", "SoLuot")
        them(so_viet(e1["seed"].nunique(), 0), "EMot", "SoSeed")
        for (bo, c), d in e1.groupby(["bo", "c"]):
            g = d[d["phuong_phap"] == "fcm_kpp"].set_index("seed")
            for pp, dp in d.groupby("phuong_phap"):
                k = ("EMot", bo, "C", c, pp)
                them(so_viet(dp["auc_rui_ro"].median(), 3), *k, "Auc")
                if "auc_cung" in dp:
                    them(so_viet(dp["auc_cung"].median(), 3), *k, "AucCung")
                them(so_viet(dp["silhouette"].median(), 3), *k, "Sil")
                them(so_viet(100 * dp["cum_nho_nhat"].median(), 1), *k, "CumNho")
                them(so_gon(dp["t_tong"].median(), 2), *k, "ThoiGian")
                them(so_gon(phan_vi(dp["XB"], [50])[0], 2, 1e4), *k, "Xb")
                if pp in CO_J:
                    them(so_viet(dp["J_m"].median(), 1), *k, "J")
                if len(g):
                    them(so_viet(dp["t_tong"].median() / g["t_tong"].median(), 1), *k, "LanFcmKpp")
                if pp in CO_J and len(g):
                    x = dp.set_index("seed")["J_m"]
                    cung = (np.abs(x - g["J_m"].loc[x.index]) <= TOL_J * g["J_m"].loc[x.index]).mean()
                    them(so_viet(100 * cung, 0), *k, "CungJ")
        dx = e1[e1["phuong_phap"] == "de_xuat"]
        if len(dx):
            them(so_viet(int((dx["goi_pso"].astype(str) == "True").sum()), 0), "EMot", "DeXuatGoiPso")
        if "fcm_kpp" in set(e1["phuong_phap"]):
            kd = kiem_dinh_e1(e1)
            them(so_viet(len(kd), 0), "EMot", "KiemDinh", "SoPhepSo")
            them(so_viet(int(kd["trung"].sum()), 0), "EMot", "KiemDinh", "SoTrung")
            them(so_viet(int((kd["p_holm"] < ALPHA).sum()), 0), "EMot", "KiemDinh", "SoKhacBiet")
            dc = doi_chieu_kiem_dinh(e1, kd)
            them(so_viet(dc["cu_trung"], 0), "EMot", "KiemDinh", "NguyenVan", "SoTrung")
            them(so_viet(dc["cu_co_p"], 0), "EMot", "KiemDinh", "NguyenVan", "SoCoP")
            them(so_viet(dc["cu_khac"], 0), "EMot", "KiemDinh", "NguyenVan", "SoKhacBiet")
            them(so_viet(dc["toan_ho_khac"], 0), "EMot", "KiemDinh", "ToanHo", "SoKhacBiet")
            them(so_viet(len(dc["o_bien"]), 0), "EMot", "KiemDinh", "SoOBien")
    if len(e2):
        for c, d in e2[e2["phuong_phap"] == "de_xuat"].groupby("c"):
            them(so_viet(int((d["goi_pso"].astype(str) == "True").sum()), 0), "EHai", "C", c, "DeXuatGoiPso")
        tl = ti_le_dat_j_tot_nhat(e2)
        for (bo, c, pp), v in tl.items():
            dp = e2[(e2["c"] == c) & (e2["phuong_phap"] == pp)]
            k = ("EHai", "C", c, pp)
            them(so_viet(100 * v, 0), *k, "DatJ")
            them(so_viet(dp["auc_rui_ro"].median(), 3), *k, "Auc")
            if "auc_cung" in dp:
                them(so_viet(dp["auc_cung"].median(), 3), *k, "AucCung")
            them(so_viet(100 * dp["cum_nho_nhat"].median(), 1), *k, "CumNho")
            them(so_gon(dp["t_tong"].median(), 2), *k, "ThoiGian")
    if len(e3):
        for _, r in thong_ke_e3(e3).iterrows():
            for kieu in ("u_ngau_nhien", "kmeans_pp"):
                k = ("EBa", r.bo, "C", r.c, kieu)
                them(str(r[f"{kieu}_so_cuc_tieu"]), *k, "SoCucTieu")
                them(so_viet(100 * r[f"{kieu}_dat_j"], 0), *k, "DatJ")
                them(so_viet(100 * r[f"{kieu}_sup"], 0), *k, "Sup")
                them(so_viet(r[f"{kieu}_pc"], 3), *k, "Pc")
    if len(e4):
        for (bo, pp), d in e4.groupby(["bo", "phuong_phap"]):
            n = d["kich_thuoc"].sum()
            for _, r in d.iterrows():
                k = ("EBon", bo, pp, "Cum", int(r.cum) + 1)
                them(so_viet(100 * r.ti_le_nhan1, 1), *k, "VoNo")
                them(so_viet(100 * r.kich_thuoc / n, 1), *k, "Co")
                them(so_viet(r["mo_ho_%"], 1), *k, "MoHo")
    return m


def ghi_so_lieu(macro: list[tuple[str, str]], duong_dan: str) -> None:
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write("% Sinh tự động bởi code/phan_tich.py — không sửa tay.\n")
        for ten, gia_tri in macro:
            f.write(f"\\newcommand{{\\{ten}}}{{{gia_tri}}}\n")


# ------------------------------------------------------------------ hình
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for t in glob.glob(os.path.join(GOC_TL, "fonts", "times*.ttf")):
        font_manager.fontManager.addfont(t)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "pdf.fonttype": 42,
                         "axes.unicode_minus": True, "axes.formatter.use_locale": False})
    return plt


def _dau_phay(ax, truc="x"):
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if truc == "x" else ax.yaxis).set_major_formatter(f)


def hinh_e1_auc(e1: pd.DataFrame, bo: str, duong_dan: str) -> None:
    plt = _plt()
    d = e1[e1["bo"] == bo]
    pp = [p for _, nhom in NHOM_HIEN_THI for p in _co_mat(d, nhom)]
    cs = sorted(d["c"].unique())
    fig, axs = plt.subplots(1, len(cs), figsize=(6.3, 4.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cs):
        dc = d[d["c"] == c]
        ax.boxplot([dc.loc[dc["phuong_phap"] == p, "auc_rui_ro"].dropna() for p in pp][::-1], vert=False,
                   widths=0.6, flierprops=dict(markersize=2))
        ax.set_title(f"c = {c}")
        ax.set_xlabel("AUC tách rủi ro")
        ax.grid(axis="x", alpha=0.3)
        _dau_phay(ax)
    np.atleast_1d(axs)[0].set_yticks(range(1, len(pp) + 1), [TEN_HIEN_THI[p] for p in pp][::-1])
    fig.tight_layout()
    fig.savefig(duong_dan)
    plt.close(fig)


def hinh_e1_thoi_gian(e1: pd.DataFrame, duong_dan: str, c: int = 3) -> None:
    plt = _plt()
    bo_ds = list(dict.fromkeys(e1["bo"]))
    fig, axs = plt.subplots(1, len(bo_ds), figsize=(6.3, 4.4), sharey=True)
    pha = [("t_khoi_tao", "Khởi tạo"), ("t_tim_kiem", "Tìm kiếm"), ("t_tinh_chinh", "Tinh chỉnh")]
    for ax, bo in zip(np.atleast_1d(axs), bo_ds):
        d = e1[(e1["bo"] == bo) & (e1["c"] == c)]
        pp = [p for _, nhom in NHOM_HIEN_THI for p in _co_mat(d, nhom)][::-1]
        tv = d.groupby("phuong_phap")[[k for k, _ in pha]].median().loc[pp]
        san = 1e-3
        trai = np.zeros(len(pp))  # thang log cắt 0 về mép trái trục, thanh không bị cộng thêm
        for k, nhan in pha:
            rong = np.where(tv[k].values > 0, tv[k].values, 0.0)
            ax.barh(range(len(pp)), rong, left=trai, label=nhan, height=0.7)
            trai = trai + rong
        ax.set_xscale("log")
        ax.set_xlim(left=san)
        ax.set_title(f"{TEN_BO.get(bo, bo)}, c = {c}")
        ax.set_xlabel("Thời gian trung vị (s, thang log)")
        ax.grid(axis="x", alpha=0.3, which="both")
        ax.set_yticks(range(len(pp)), [TEN_HIEN_THI[p] for p in pp])
    h, n = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, n, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(duong_dan)
    plt.close(fig)


def hinh_hoi_tu(thu_muc: str, e1: pd.DataFrame, bo: str, duong_dan: str) -> bool:
    plt = _plt()
    cs = sorted(e1.loc[e1["bo"] == bo, "c"].unique())
    fig, axs = plt.subplots(1, len(cs), figsize=(6.3, 2.8), sharey=False)
    co_net = False
    for ax, c in zip(np.atleast_1d(axs), cs):
        g = e1[(e1["bo"] == bo) & (e1["c"] == c) & (e1["phuong_phap"] == "fcm_kpp") & (e1["seed"] == 1000)]
        if g.empty:
            continue
        j_goc = float(g["J_m"].iloc[0])
        for pp in [p for p in CO_LICH_SU if p in HIEN_THI]:
            t = os.path.join(thu_muc, "hoi-tu", f"E1_{bo}_c{c}_{pp}.json")
            if os.path.exists(t):
                with open(t) as f:
                    ls = np.asarray(json.load(f))
                ax.plot(np.arange(len(ls)), ls / j_goc, label=TEN_HIEN_THI[pp], lw=1,
                        color=plt.get_cmap("tab10")(CO_LICH_SU.index(pp)))
                co_net = True
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.set_title(f"c = {c}")
        ax.set_xlabel("Vòng")
        ax.grid(alpha=0.3)
        _dau_phay(ax, "y")
    np.atleast_1d(axs)[0].set_ylabel("gbest / $J_m$ (FCM k-means++)")
    nhan = {}
    for ax in np.atleast_1d(axs):
        nhan.update(zip(*ax.get_legend_handles_labels()[::-1]))
    fig.legend(list(nhan.values()), list(nhan), loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(duong_dan)
    plt.close(fig)
    return co_net


# ------------------------------------------------------------------ chạy
def chay(thu_muc: str, hinh: str) -> list[str]:
    bang = os.path.join(thu_muc, "bang")
    os.makedirs(bang, exist_ok=True)
    os.makedirs(hinh, exist_ok=True)
    ra = []

    def ghi(ten, noi_dung):
        with open(os.path.join(bang, ten), "w", encoding="utf-8") as f:
            f.write(noi_dung)
        ra.append(os.path.join(bang, ten))

    mo_ta = _tep_mo_ta(thu_muc)
    if mo_ta:
        ghi("mo-ta-du-lieu.tex", bang_mo_ta(pd.read_csv(mo_ta)))
    e1, e2, e3, e4 = (_doc(thu_muc, e) for e in ("E1", "E2", "E3", "E4"))
    if len(e1):
        for bo in dict.fromkeys(e1["bo"]):
            ghi(f"e1-ket-qua-{bo}.tex", bang_e1_ket_qua(e1, bo))
            ghi(f"e1-thoi-gian-{bo}.tex", bang_e1_thoi_gian(e1, bo))
            hinh_e1_auc(e1, bo, os.path.join(hinh, f"e1-auc-{bo}.pdf"))
            ra.append(os.path.join(hinh, f"e1-auc-{bo}.pdf"))
            if hinh_hoi_tu(thu_muc, e1, bo, os.path.join(hinh, f"hoi-tu-{bo}.pdf")):
                ra.append(os.path.join(hinh, f"hoi-tu-{bo}.pdf"))
        kd = kiem_dinh_e1(e1)
        ghi("e1-kiem-dinh.tex", bang_e1_kiem_dinh(kd, doi_chieu_kiem_dinh(e1, kd)))
        hinh_e1_thoi_gian(e1, os.path.join(hinh, "e1-thoi-gian.pdf"))
        ra.append(os.path.join(hinh, "e1-thoi-gian.pdf"))
    if len(e2):
        ghi("e2-it-nhieu-cum.tex", bang_e2(e2))
    if len(e3):
        ghi("e3-chan-doan.tex", bang_e3(e3))
    if len(e4):
        import tien_xu_ly as tx

        for bo in dict.fromkeys(e4["bo"]):
            do_lech = pd.Series(tx.nap(bo).X_goc.std(axis=0), index=tx.nap(bo).ten_cot)
            ghi(f"e4-profile-{bo}.tex", bang_e4(e4, bo, chon_thuoc_tinh(e4, bo, do_lech)))
    ghi_so_lieu(macro_so_lieu(thu_muc), os.path.join(thu_muc, "so-lieu.tex"))
    ra.append(os.path.join(thu_muc, "so-lieu.tex"))
    return ra


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--hinh", default=HINH_MAC_DINH)
    a = ap.parse_args(argv)
    for t in chay(os.path.abspath(a.ket_qua), os.path.abspath(a.hinh)):
        print(t)


if __name__ == "__main__":
    main()
