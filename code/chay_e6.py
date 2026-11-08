# -*- coding: utf-8 -*-
"""Thí nghiệm E6: kết luận của E5 có phụ thuộc số mũ mờ m = 2 và số cụm c không.

    python3 chay_e6.py [--tien-trinh 12] [--ket-qua DIR] [--chi-phan-tich]

Hai bộ chính × m ∈ {1,5; 2} × c = 2..8 × 30 seed; mỗi lượt chạy FCM từ 4 kiểu khởi tạo: U ngẫu nhiên, c điểm ngẫu
nhiên, k-means++ và PSO-W (20 hạt, 30 vòng). Ghi <ket-qua>/E6/e6.csv (chạy lại chỉ chạy phần thiếu), rồi
<ket-qua>/bang/e6-cuc-tieu.tex và <ket-qua>/so-lieu-e6.tex.
"""

from __future__ import annotations

import os

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

import argparse  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from chay_thuc_nghiem import KET_QUA_MAC_DINH, NGUONG_SUP, SEED_DAU, SO_SEED, nap  # noqa: E402
from fcm import FCM  # noqa: E402
from giao_dien import fcm_kpp, fcm_mot_lan  # noqa: E402
from phan_tich import (TEN_BO, TOL_J, _bang_float, ghi_so_lieu, hai_dong, so_cuc_tieu, so_gon, so_viet,  # noqa: E402
                       ten_macro, tex, wilson, mcnemar_cap)
from pso_khoi_tao import fcm_diem_ngau_nhien, ham_e5  # noqa: E402

BO = ("viet_nam", "taiwan_kiem_chung")
M = (1.5, 2.0)
C = tuple(range(2, 9))
KIEU = {"fcm": fcm_mot_lan, "fcm_diem": fcm_diem_ngau_nhien, "fcm_kpp": fcm_kpp, "pso_w": ham_e5("w", 20, 30)}
TEN_KIEU = {"fcm": ("U ngẫu", "nhiên"), "fcm_diem": ("Điểm", "ngẫu nhiên"), "fcm_kpp": ("k-", "means++"),
            "pso_w": ("PSO-W", "")}
KHOA = ["bo", "m", "c", "seed", "kieu"]


def auc_rui_ro(U, y):
    nhan = U.argmax(1)
    ti_le = [y[nhan == k].mean() if (nhan == k).any() else -1.0 for k in range(U.shape[1])]
    return float(roc_auc_score(y, U[:, int(np.argmax(ti_le))]))


def lam(tac_vu):
    bo, m, c, seed = tac_vu
    d = nap(bo)
    dong = []
    for kieu, ham in KIEU.items():
        kq = ham(d.X, c, m, np.random.default_rng(seed), None)
        V = kq.V
        kc = np.sqrt(((V[:, None] - V[None]) ** 2).sum(-1))[np.triu_indices(c, 1)].min()
        dong.append(dict(bo=bo, m=m, c=c, seed=seed, kieu=kieu, J=FCM(c, m=m).objective_from_centroids(d.X, V),
                         sup=bool(np.abs(V).max() < NGUONG_SUP), kc_tam_min=float(kc),
                         PC=float((kq.U ** 2).sum() / len(d.X)), auc=auc_rui_ro(kq.U, d.y),
                         vong_fcm=int(kq.n_vong_fcm), t=kq.t_tong))
    return dong


def chay(thu_muc: str, so_tien_trinh: int) -> pd.DataFrame:
    tep = os.path.join(thu_muc, "E6", "e6.csv")
    os.makedirs(os.path.dirname(tep), exist_ok=True)
    cu = pd.read_csv(tep) if os.path.exists(tep) else pd.DataFrame(columns=KHOA)
    xong = set(zip(cu["bo"], cu["m"].astype(float), cu["c"].astype(int), cu["seed"].astype(int)))
    viec = [(b, m, c, SEED_DAU + s) for b in BO for m in M for c in C for s in range(SO_SEED)
            if (b, m, c, SEED_DAU + s) not in xong]
    if viec:
        with Pool(so_tien_trinh) as p:
            moi = [r for ds in p.imap_unordered(lam, viec, chunksize=2) for r in ds]
        cu = pd.concat([cu, pd.DataFrame(moi)], ignore_index=True) if len(cu) else pd.DataFrame(moi)
        cu.sort_values(KHOA).to_csv(tep, index=False)
    print(f"E6: chạy {len(viec)} tác vụ, tổng {len(cu)} dòng")
    return cu


def tom_tat(d: pd.DataFrame) -> pd.DataFrame:
    ra = []
    for (b, m, c), g in d.groupby(["bo", "m", "c"]):
        j_min = g["J"].min()
        r = dict(bo=b, m=m, c=int(c), so_muc=so_cuc_tieu(g["J"]), sup=100 * g["sup"].mean(),
                 kc_tam=g["kc_tam_min"].median(), pc=g["PC"].median(), auc=g["auc"].median())
        dat = {}
        for k in KIEU:
            gk = g[g["kieu"] == k]
            dat[k] = gk["J"] <= (1 + TOL_J) * j_min
            r[f"dat_{k}"] = 100 * dat[k].mean()
            r[f"t_{k}"] = gk["t"].median()
        for k in KIEU:
            r[f"k_{k}"], r[f"n_{k}"] = int(dat[k].sum()), int(len(dat[k]))
        theo_seed = {k: pd.Series(dat[k].to_numpy(), index=g.loc[g["kieu"] == k, "seed"].to_numpy()) for k in KIEU}
        r["p_mcnemar"] = mcnemar_cap(theo_seed["pso_w"], theo_seed["fcm_kpp"])[2]
        ra.append(r)
    return pd.DataFrame(ra)


def bang(tt: pd.DataFrame) -> str:
    than = []
    for i, ((b, m), g) in enumerate(tt.groupby(["bo", "m"], sort=False)):
        if i:
            than.append("\\midrule")
        for _, r in g.iterrows():
            nhan = f"{tex(TEN_BO.get(b, b))}, m = {so_viet(m, 1)}" if r.c == g["c"].min() else ""
            than.append(" & ".join([nhan, str(r.c), str(r.so_muc), *[so_viet(r[f"dat_{k}"], 0) for k in KIEU],
                                    so_viet(r.sup, 0), so_gon(r.kc_tam, 2), so_viet(r.auc, 3)]) + " \\\\")
    tieu_de = [" & ".join(["Bộ, m", "c", hai_dong("Số", "cực tiểu"),
                           *[hai_dong(*TEN_KIEU[k]) for k in KIEU], hai_dong("Sụp", "(\\%)"),
                           hai_dong("K/c tâm", "nhỏ nhất"), "AUC"]) + " \\\\"]
    return _bang_float("@{}lrrrrrrrrr@{}", tieu_de, than,
                       "TN2 -- số cực tiểu của J\\textsubscript{m} và tỉ lệ đạt J tốt nhất theo m, c và kiểu khởi tạo",
                       "tab:e6-cuc-tieu",
                       "30 seed mỗi kiểu khởi tạo; số cực tiểu: số mức J\\textsubscript{m} lệch nhau quá $10^{-4}$ "
                       "(tương đối) trên 120 lượt; 4 cột kiểu khởi tạo: tỉ lệ lượt (\\%) có J\\textsubscript{m} "
                       "$\\le (1 + 10^{-4})$ lần mức thấp nhất; "
                       "K/c tâm nhỏ nhất: khoảng cách nhỏ nhất giữa 2 tâm (dữ liệu z-score), trung vị; AUC: AUC tách "
                       "rủi ro, trung vị. PSO-W: 20 hạt, 30 vòng.")


def macro(tt: pd.DataFrame) -> list[tuple[str, str]]:
    ra = [("soLieuESauSoCauHinh", str(len(tt))), ("soLieuESauSoCauHinhMotCucTieu", str(int((tt["so_muc"] == 1).sum())))]
    for _, r in tt.iterrows():
        goc = ["ESau", r.bo, "M", so_viet(r.m, 1).replace(",", "Phay"), "C", r.c]
        ra += [("soLieu" + ten_macro(*goc, "SoMuc"), str(r.so_muc)), ("soLieu" + ten_macro(*goc, "Auc"), so_viet(r.auc, 3)),
               ("soLieu" + ten_macro(*goc, "KcTam"), so_gon(r.kc_tam, 2))]
        ra += [("soLieu" + ten_macro(*goc, k, "Dat"), so_viet(r[f"dat_{k}"], 0)) for k in KIEU]
        ra += [("soLieu" + ten_macro(*goc, k, "ThoiGian"), so_gon(r[f"t_{k}"], 2)) for k in ("fcm_kpp", "pso_w")]
        if r.so_muc > 1:
            ra.append(("soLieu" + ten_macro(*goc, "McP"), so_viet(r.p_mcnemar, 2)))
            for k in KIEU:
                lo, hi = wilson(r[f"k_{k}"], r[f"n_{k}"])
                ra += [("soLieu" + ten_macro(*goc, k, "K"), str(r[f"k_{k}"])),
                       ("soLieu" + ten_macro(*goc, k, "WilsonDuoi"), so_viet(lo, 1)),
                       ("soLieu" + ten_macro(*goc, k, "WilsonTren"), so_viet(hi, 1))]
    for (b, m), g in tt.groupby(["bo", "m"]):
        goc = ["ESau", b, "M", so_viet(m, 1).replace(",", "Phay")]
        ra += [("soLieu" + ten_macro(*goc, "AucMax"), so_viet(g["auc"].max(), 3)),
               ("soLieu" + ten_macro(*goc, "AucMin"), so_viet(g["auc"].min(), 3))]
    return ra


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tien-trinh", type=int, default=12)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    ap.add_argument("--chi-phan-tich", action="store_true")
    a = ap.parse_args(argv)
    thu_muc = os.path.abspath(a.ket_qua)
    d = pd.read_csv(os.path.join(thu_muc, "E6", "e6.csv")) if a.chi_phan_tich else chay(thu_muc, a.tien_trinh)
    tt = tom_tat(d)
    os.makedirs(os.path.join(thu_muc, "bang"), exist_ok=True)
    with open(os.path.join(thu_muc, "bang", "e6-cuc-tieu.tex"), "w", encoding="utf-8") as f:
        f.write(bang(tt))
    ghi_so_lieu(macro(tt), os.path.join(thu_muc, "so-lieu-e6.tex"))
    print(tt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
