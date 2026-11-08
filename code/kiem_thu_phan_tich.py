# -*- coding: utf-8 -*-
"""Kiểm thử phần phân tích (thống kê, định dạng, sinh bảng LaTeX). Chạy: python3 kiem_thu_phan_tich.py"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

import phan_tich as pt
from danh_gia import COT

GOC_TL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _e1_gia(phuong_phap=("kmeans", "fcm_kpp", "de_xuat", "c_msgwo_wfcm", "cach7_pso_trong_so", "wfcm_giai_tich",
                         "kfcm", "cach8_pso_kfcm", "a_S"), so_seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dong = []
    for c in (2, 3):
        for pp in phuong_phap:
            for s in range(so_seed):
                d = {k: float(rng.random()) for k in COT}
                d.update(thi_nghiem="E1", bo="taiwan_kiem_chung", c=c, seed=1000 + s, phuong_phap=pp, goi_pso="",
                         J_m=1234.5678 + s, loi="")
                dong.append(d)
    return pd.DataFrame(dong)


def test_holm_vi_du_biet_truoc():
    assert np.allclose(pt.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(pt.holm([0.5, 0.9]), [1.0, 1.0])


def test_wilcoxon_moi_hieu_bang_0_tra_trung():
    a = np.array([0.7, 0.71, 0.72])
    assert pt.wilcoxon_cap(a, a.copy()) == "trùng"
    p = pt.wilcoxon_cap(np.arange(10.0), np.arange(10.0) + 1)
    assert isinstance(p, float) and 0 < p < 0.05


def test_kiem_dinh_cung_J_la_trung_du_AUC_lech_so_hoc():
    seed = np.arange(1000, 1030)
    goc = pd.DataFrame(dict(bo="b", c=3, seed=seed, phuong_phap="fcm_kpp", J_m=100.0, auc_rui_ro=0.6))
    cung = goc.assign(phuong_phap="cach2_fcm_gieo", J_m=100.0 * (1 + 1e-7), auc_rui_ro=0.6 + 1e-6)
    loai_ii = goc.assign(phuong_phap="kfcm", J_m=100.0, auc_rui_ro=0.6 + 1e-6)
    kd = pt.kiem_dinh_e1(pd.concat([goc, cung, loai_ii])).set_index("phuong_phap")
    assert kd.loc["cach2_fcm_gieo", "trung"] and kd.loc["cach2_fcm_gieo", "hieu_tv"] == 0
    assert not kd.loc["kfcm", "trung"], "loại II không so J_m: hiệu AUC giữ nguyên"
    cu = pt.kiem_dinh_e1(pd.concat([goc, cung, loai_ii]), gom_cung_j=False).set_index("phuong_phap")
    assert not cu.loc["cach2_fcm_gieo", "trung"], "quy tắc cũ (so 0 tuyệt đối) để báo song song"


def test_holm_toan_ho_tinh_trung_la_p_bang_1():
    kd = pd.DataFrame(dict(p=[0.01, np.nan, 0.02], trung=[False, True, False]))
    assert np.allclose(pt.holm_toan_ho(kd), [0.03, 1.0, 0.04])


def test_phan_vi_chung_cho_bang_va_macro_khi_co_vo_cung():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, np.inf, np.inf])
    q1, tv, q3 = pt.phan_vi(s)
    assert tv == 3.5 and q1 == 2.25 and q3 == np.inf
    assert pt.phan_vi(pd.Series([1.0, np.inf]))[1] == np.inf
    assert pt.phan_vi(pd.Series([1.0, 2.0, np.nan]))[1] == 1.5


def test_dinh_dang_so_viet():
    assert pt.so_viet(1234.5678) == "1.234,57"
    assert pt.so_viet(0.5) == "0,50"
    assert pt.so_viet(-1234567.0, 0) == "−1.234.567"
    assert pt.so_viet(float("nan")) == "--"


def test_ten_macro_hop_le():
    with tempfile.TemporaryDirectory() as tmp:
        for e in ("E1",):
            os.makedirs(os.path.join(tmp, e))
        _e1_gia().to_csv(os.path.join(tmp, "E1", "taiwan_kiem_chung_c2.csv"), index=False)
        macro = pt.macro_so_lieu(tmp)
    ten = [t for t, _ in macro]
    assert ten and all(re.fullmatch(r"soLieu[A-Za-z]+", t) for t in ten), [t for t in ten if not t.isalpha()][:5]
    assert len(ten) == len(set(ten)), "tên macro trùng"
    assert all(re.fullmatch(r"[A-Za-z]+", pt.ten_macro(p)) for p in pt.TEN_HIEN_THI)


def test_bang_tex_bien_dich_duoc():
    if not shutil.which("xelatex"):
        raise AssertionError("cần xelatex để kiểm tra bảng biên dịch được")
    bang = pt.bang_e1_ket_qua(_e1_gia(), "taiwan_kiem_chung")
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "bang.tex"), "w") as f:
            f.write(bang)
        with open(os.path.join(tmp, "thu.tex"), "w") as f:
            f.write("\\documentclass[a4paper,12pt]{report}\n\\input{noi-dung/preamble}\n\\begin{document}\n"
                    f"\\input{{{tmp}/bang.tex}}\n\\end{{document}}\n")
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}",
                            os.path.join(tmp, "thu.tex")], cwd=GOC_TL, capture_output=True, text=True, timeout=300)
        log = open(os.path.join(tmp, "thu.log"), encoding="utf-8", errors="replace").read()
    assert r.returncode == 0 and "! " not in log, log[-3000:]
    assert "Overfull \\hbox" not in log, re.findall(r"Overfull \\hbox.*", log)[:3]


def test_bang_e1_chi_hien_phuong_phap_trinh_bay():
    bang = pt.bang_e1_ket_qua(_e1_gia(), "taiwan_kiem_chung")
    tieu_de = next(d for d in bang.splitlines() if d.startswith("Phương pháp &"))
    assert "cứng" not in tieu_de and tieu_de.count("&") == 7, tieu_de
    for pp in ("cach7_pso_trong_so", "wfcm_giai_tich", "kfcm", "cach8_pso_kfcm"):
        assert not any(d.startswith(pt.TEN_HIEN_THI[pp] + " &") for d in bang.splitlines()), pp


def test_J_m_an_voi_loai_II():
    bang = pt.bang_e1_ket_qua(_e1_gia(), "taiwan_kiem_chung")
    dong = {pp: next(d for d in bang.splitlines() if d.startswith(pt.TEN_HIEN_THI[pp] + " &"))
            for pp in ("c_msgwo_wfcm", "fcm_kpp", "a_S")}
    for pp in ("c_msgwo_wfcm",):
        assert "1.23" not in dong[pp] and dong[pp].split("&")[1].strip() == "--", dong[pp]
    for pp in ("fcm_kpp", "a_S"):
        assert "1.23" in dong[pp].split("&")[1], dong[pp]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ĐẠT   {name}")
        except Exception as exc:
            failed += 1
            print(f"LỖI   {name}: {exc!r}"[:1500])
    print(f"\n{len(tests) - failed}/{len(tests)} phép thử đạt")
    raise SystemExit(1 if failed else 0)
