# -*- coding: utf-8 -*-
"""Trình chạy thực nghiệm chính thức E1–E4 (thiết kế và tiêu chí: mục 3.2 của tiểu luận).

    python3 chay_thuc_nghiem.py --thi-nghiem E1|E2|E3|E4|tat-ca [--nhanh] [--tien-trinh 8] [--ket-qua DIR]

- E1, E2: một tác vụ = (bộ, c, seed), chạy TUẦN TỰ mọi phương pháp để thời gian đo cùng điều kiện máy; mỗi phương pháp
  nhận default_rng(seed) riêng nên thứ tự chạy không đổi kết quả. Một dòng mỗi (phương pháp, seed) trong
  <ket-qua>/<E>/<bo>_c<c>.csv. Seed 1000: lịch sử fitness ghi vào hoi-tu/<E>_<bo>_c<c>_<phuong_phap>.json.
- E3: một tác vụ = (bộ, c); FCM từ SO_KHOI_TAO khởi tạo U ngẫu nhiên và SO_KHOI_TAO khởi tạo k-means++ (seed 1000 + i).
- E4: một tác vụ = (bộ, phương pháp), c = 3, seed 1000; profile cụm trên giá trị gốc X_goc.
Song song giữa các tác vụ bằng multiprocessing (1 luồng BLAS mỗi tiến trình); worker không ghi file, tiến trình chính
ghi nối tiếp và bỏ qua khoá đã có, nên chạy lại cùng lệnh sẽ chạy tiếp phần thiếu. --nhanh: 3 seed, E3 5 khởi tạo.
"""

from __future__ import annotations

import os

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from functools import lru_cache  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cach_ket_hop as ck  # noqa: E402
import tien_xu_ly as tx  # noqa: E402
from danh_gia import COT, danh_gia, mau_silhouette  # noqa: E402
from de_xuat import de_xuat_ep_pso, pso_fcm_chan_doan  # noqa: E402
from fcm import FCM  # noqa: E402
from giao_dien import fcm_kpp, fcm_mot_lan, kmeans  # noqa: E402
from pso_trong_so import pso_trong_so  # noqa: E402
from thuat_toan_a import a_L, a_LOG, a_S  # noqa: E402
from thuat_toan_b import qpso_fcm  # noqa: E402
from thuat_toan_c import msgwo_wfcm  # noqa: E402

KET_QUA_MAC_DINH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M = 2.0
SEED_DAU, SO_SEED, SO_SEED_NHANH = 1000, 30, 3
SO_KHOI_TAO, SO_KHOI_TAO_NHANH = 30, 5
NGUONG_SUP = 1e-3  # |V|max < ngưỡng: mọi tâm dồn về trung bình dữ liệu (z-score) — FCM sụp
NGUONG_MO_HO = 0.6

HAM = {
    "kmeans": kmeans,
    "fcm": fcm_mot_lan,
    "fcm_kpp": fcm_kpp,
    "fcm_lap_lai": ck.fcm_lap_lai,
    "a_L": a_L,
    "a_S": a_S,
    "a_LOG": a_LOG,
    "b_qpso_fcm": qpso_fcm,
    "c_msgwo_wfcm": msgwo_wfcm,
    "cach2_fcm_gieo": ck.cach2_fcm_gieo,
    "cach3_xen_ke": ck.cach3_xen_ke,
    "cach5_pso_U": ck.cach5_pso_U,
    "wfcm_giai_tich": ck.wfcm_giai_tich,
    "cach7_pso_trong_so": ck.cach7_pso_trong_so,
    "kfcm": ck.kfcm,
    "cach8_pso_kfcm": ck.cach8_pso_kfcm,
    "de_xuat": pso_fcm_chan_doan,
    "de_xuat_ep_pso": de_xuat_ep_pso,
    "pso_trong_so": pso_trong_so,
}
PHUONG_PHAP = {
    "E1": ["kmeans", "fcm", "fcm_kpp", "fcm_lap_lai", "a_L", "a_S", "a_LOG", "b_qpso_fcm", "c_msgwo_wfcm",
           "cach2_fcm_gieo", "cach3_xen_ke", "cach5_pso_U", "wfcm_giai_tich", "cach7_pso_trong_so", "kfcm",
           "cach8_pso_kfcm", "de_xuat"],
    "E2": ["kmeans", "fcm", "fcm_kpp", "fcm_lap_lai", "a_LOG", "b_qpso_fcm", "cach2_fcm_gieo", "cach3_xen_ke",
           "de_xuat", "de_xuat_ep_pso"],
    "E4": ["fcm_kpp", "kmeans", "cach7_pso_trong_so", "pso_trong_so"],
}
CAU_HINH = {
    "E1": [(bo, c) for bo in ("viet_nam", "taiwan_kiem_chung") for c in (2, 3, 4)],
    "E2": [("taiwan_kiem_chung_khong_log", c) for c in (3, 6)],
    "E3": [(bo, c) for bo in ("viet_nam", "taiwan_kiem_chung", "taiwan_kiem_chung_khong_log", "south_german",
                              "german_so", "australian", "polish", "taiwan_pha_san")
           for c in ((2, 3, 4, 5, 6) if bo.startswith("taiwan_kiem_chung") else (2, 3, 4))],
    "E4": [(bo, 3) for bo in ("viet_nam", "taiwan_kiem_chung")],
}
THI_NGHIEM = ("E1", "E2", "E3", "E4")
KHOA = ["thi_nghiem", "bo", "c", "seed", "phuong_phap"]
BO_THEM: dict = {}  # bộ dữ liệu bổ sung cho phép thử (tên → hàm trả BoDuLieu)


class TacVu(NamedTuple):
    thi_nghiem: str
    bo: str
    c: int
    khoa: object  # E1/E2: seed; E3: số khởi tạo mỗi kiểu; E4: tên phương pháp


@lru_cache(maxsize=None)
def nap(ten: str) -> tx.BoDuLieu:
    return BO_THEM[ten]() if ten in BO_THEM else tx.nap(ten)


def danh_sach_tac_vu(thi_nghiem: str, nhanh: bool = False) -> list[TacVu]:
    if thi_nghiem in ("E1", "E2"):
        so_seed = SO_SEED_NHANH if nhanh else SO_SEED
        return [TacVu(thi_nghiem, bo, c, SEED_DAU + s) for bo, c in CAU_HINH[thi_nghiem] for s in range(so_seed)]
    if thi_nghiem == "E3":
        return [TacVu("E3", bo, c, SO_KHOI_TAO_NHANH if nhanh else SO_KHOI_TAO) for bo, c in CAU_HINH["E3"]]
    if thi_nghiem == "E4":
        return [TacVu("E4", bo, c, pp) for bo, c in CAU_HINH["E4"] for pp in PHUONG_PHAP["E4"]]
    raise KeyError(thi_nghiem)


# ------------------------------------------------------------------ E1, E2
def chay_tac_vu(tv: TacVu, phuong_phap: list[str] | None = None) -> tuple[list[dict], dict[str, list[float]]]:
    """Chạy tuần tự các phương pháp trên (bộ, c, seed); trả (các dòng, lịch sử fitness theo phương pháp)."""
    bo = nap(tv.bo)
    mau = mau_silhouette(len(bo.X))
    dong, lich_su = [], {}
    for ten in phuong_phap or PHUONG_PHAP[tv.thi_nghiem]:
        khoa = dict(thi_nghiem=tv.thi_nghiem, bo=tv.bo, c=tv.c, seed=tv.khoa, phuong_phap=ten)
        try:
            kq = HAM[ten](bo.X, tv.c, M, np.random.default_rng(tv.khoa), None)
            dong.append({**khoa, **danh_gia(bo.X, bo.y, kq, M, mau), "loi": ""})
            if kq.lich_su:
                lich_su[ten] = [float(v) for v in kq.lich_su]
        except Exception as exc:  # một phương pháp lỗi không dừng cả tác vụ; phase kiểm tra soát cột loi
            dong.append({**khoa, **{k: np.nan for k in COT}, "loi": repr(exc)})
    return dong, lich_su


# ------------------------------------------------------------------ E3
def chay_e3(tv: TacVu) -> list[dict]:
    bo = nap(tv.bo)
    X, mo_hinh = bo.X, FCM(tv.c, m=M)
    dong = []
    for kieu in ("u_ngau_nhien", "kmeans_pp"):
        for i in range(tv.khoa):
            rng = np.random.default_rng(SEED_DAU + i)
            t0 = time.perf_counter()
            if kieu == "u_ngau_nhien":
                U, V, n_vong = mo_hinh.fit(X, seed=rng)
            else:
                U, V, n_vong = mo_hinh.fit(X, init_v=mo_hinh.init_centroids_kmeans_plusplus(X, seed=rng), with_u=False)
            dong.append(dict(thi_nghiem="E3", bo=tv.bo, c=tv.c, kieu=kieu, i=i, seed=SEED_DAU + i,
                             J_m=mo_hinh.objective_from_centroids(X, V), PC=float(np.mean(np.sum(U**2, axis=1))),
                             sup=int(np.abs(V).max() < NGUONG_SUP), n_vong=n_vong, t=time.perf_counter() - t0))
    return dong


# ------------------------------------------------------------------ E4
def chay_e4(tv: TacVu) -> list[dict]:
    """Profile cụm cứng: kích thước, tỉ lệ nhãn 1, % điểm mơ hồ (max U < NGUONG_MO_HO), trung bình X_goc mỗi thuộc
    tính. Cụm đánh số lại theo tỉ lệ nhãn 1 tăng dần để bảng các phương pháp đọc cùng chiều."""
    bo = nap(tv.bo)
    kq = HAM[tv.khoa](bo.X, tv.c, M, np.random.default_rng(SEED_DAU), None)
    nhan, mo_ho = kq.nhan, kq.U.max(axis=1) < NGUONG_MO_HO
    dong = []
    for k in range(tv.c):
        trong = nhan == k
        if not trong.any():
            continue
        dong.append(dict(thi_nghiem="E4", bo=tv.bo, c=tv.c, phuong_phap=tv.khoa, cum_goc=k,
                         kich_thuoc=int(trong.sum()), ti_le_nhan1=float(bo.y[trong].mean()),
                         **{"mo_ho_%": 100.0 * float(mo_ho[trong].mean())},
                         **dict(zip(bo.ten_cot, bo.X_goc[trong].mean(axis=0).tolist()))))
    dong.sort(key=lambda d: d["ti_le_nhan1"])
    for i, d in enumerate(dong):
        d["cum"] = i
    return dong


# ------------------------------------------------------------------ điều phối và ghi
def _duong_dan(thu_muc: str, tv: TacVu) -> str:
    ten = f"{tv.bo}_{tv.khoa}.csv" if tv.thi_nghiem == "E4" else f"{tv.bo}_c{tv.c}.csv"
    return os.path.join(thu_muc, tv.thi_nghiem, ten)


def _doc(duong_dan: str) -> pd.DataFrame | None:
    return pd.read_csv(duong_dan) if os.path.exists(duong_dan) else None


def _khong_loi(df: pd.DataFrame) -> pd.Series:
    return df["loi"].isna() | (df["loi"].astype(str) == "")


def _con_thieu(thu_muc: str, tv: TacVu) -> list[str] | None:
    """Phần còn phải chạy của tác vụ; None = đã đủ."""
    df = _doc(_duong_dan(thu_muc, tv))
    if tv.thi_nghiem in ("E1", "E2"):
        co = set() if df is None else set(df.loc[(df["seed"] == tv.khoa) & _khong_loi(df), "phuong_phap"])
        thieu = [p for p in PHUONG_PHAP[tv.thi_nghiem] if p not in co]
        return thieu or None
    if tv.thi_nghiem == "E3":
        return None if df is not None and len(df) >= 2 * tv.khoa else []
    return None if df is not None else []


def _lam(viec: tuple[TacVu, list[str]]):
    tv, thieu = viec
    t0 = time.perf_counter()
    if tv.thi_nghiem in ("E1", "E2"):
        dong, lich_su = chay_tac_vu(tv, thieu)
    else:
        dong, lich_su = (chay_e3(tv) if tv.thi_nghiem == "E3" else chay_e4(tv)), {}
    return tv, dong, lich_su, time.perf_counter() - t0


def _ghi(thu_muc: str, tv: TacVu, dong: list[dict], lich_su: dict) -> None:
    duong_dan = _duong_dan(thu_muc, tv)
    os.makedirs(os.path.dirname(duong_dan), exist_ok=True)
    df = pd.DataFrame(dong)
    if tv.thi_nghiem == "E3" and os.path.exists(duong_dan):
        os.remove(duong_dan)  # E3 ghi trọn một (bộ, c); phần dở dang (nếu có) chạy lại toàn bộ
    cu = _doc(duong_dan) if tv.thi_nghiem in ("E1", "E2") else None
    if cu is not None and not _khong_loi(cu).all():  # dòng lỗi cũ được thay bằng lượt chạy lại
        thay = cu["seed"].eq(tv.khoa) & cu["phuong_phap"].isin(df["phuong_phap"])
        pd.concat([cu[~thay], df], ignore_index=True).to_csv(duong_dan, index=False)
    else:
        df.to_csv(duong_dan, mode="a", header=not os.path.exists(duong_dan), index=False)
    _ghi_hoi_tu(thu_muc, tv, lich_su)


def _ghi_hoi_tu(thu_muc: str, tv: TacVu, lich_su: dict) -> None:
    if tv.thi_nghiem in ("E1", "E2") and tv.khoa == SEED_DAU:
        os.makedirs(os.path.join(thu_muc, "hoi-tu"), exist_ok=True)
        for ten, ls in lich_su.items():
            with open(os.path.join(thu_muc, "hoi-tu", f"{tv.thi_nghiem}_{tv.bo}_c{tv.c}_{ten}.json"), "w") as f:
                json.dump(ls, f)


def chay_va_ghi(tac_vu: list[TacVu], thu_muc: str, so_tien_trinh: int = 8, nhat_ky=None) -> int:
    """Chạy các tác vụ còn thiếu, ghi kết quả; trả số tác vụ đã chạy."""
    viec = [(tv, thieu) for tv in tac_vu if (thieu := _con_thieu(thu_muc, tv)) is not None]
    # tác vụ nặng trước để cân tải: Việt Nam (FCM hội tụ chậm), bộ nhiều chiều
    viec.sort(key=lambda v: (v[0].bo not in ("viet_nam", "polish", "taiwan_pha_san"),))
    if so_tien_trinh <= 1:
        ket_qua = map(_lam, viec)
        pool = None
    else:
        pool = Pool(so_tien_trinh)
        ket_qua = pool.imap_unordered(_lam, viec, chunksize=1)
    try:
        for i, (tv, dong, lich_su, t) in enumerate(ket_qua, 1):
            _ghi(thu_muc, tv, dong, lich_su)
            if nhat_ky:
                loi = sum(1 for d in dong if d.get("loi"))
                print(f"[{i}/{len(viec)}] {tv.thi_nghiem} {tv.bo} c={tv.c} {tv.khoa}: {len(dong)} dòng, "
                      f"{t:.1f} s{f', {loi} LỖI' if loi else ''}", file=nhat_ky, flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return len(viec)


def _moi_truong() -> str:
    import scipy
    import sklearn

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(d.split(":", 1)[1].strip() for d in f if d.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return (f"CPU: {cpu}\nSố lõi logic: {os.cpu_count()}\nHệ điều hành: {platform.platform()}\n"
            f"Python: {platform.python_version()}\nnumpy: {np.__version__}\nscipy: {scipy.__version__}\n"
            f"pandas: {pd.__version__}\nscikit-learn: {sklearn.__version__}\n"
            f"Luồng BLAS mỗi tiến trình: {os.environ.get('OPENBLAS_NUM_THREADS')}\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--thi-nghiem", required=True, choices=THI_NGHIEM + ("tat-ca",))
    ap.add_argument("--nhanh", action="store_true")
    ap.add_argument("--tien-trinh", type=int, default=8)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    a = ap.parse_args(argv)
    thu_muc = os.path.abspath(a.ket_qua)
    os.makedirs(thu_muc, exist_ok=True)
    ds = THI_NGHIEM if a.thi_nghiem == "tat-ca" else (a.thi_nghiem,)
    with open(os.path.join(thu_muc, "moi-truong.txt"), "a") as f:
        f.write(f"=== Lần chạy {datetime.now().isoformat(timespec='seconds')}: {' '.join(sys.argv[1:])}\n"
                f"{_moi_truong()}Số tiến trình: {a.tien_trinh}\n")
    for e in ds:
        t0 = time.perf_counter()
        print(f"== {e} bắt đầu {datetime.now():%H:%M:%S}", flush=True)
        n = chay_va_ghi(danh_sach_tac_vu(e, a.nhanh), thu_muc, a.tien_trinh, nhat_ky=sys.stdout)
        t = time.perf_counter() - t0
        print(f"== {e} xong: {n} tác vụ, {t:.0f} s", flush=True)
        with open(os.path.join(thu_muc, "moi-truong.txt"), "a") as f:
            f.write(f"{e}: {n} tác vụ chạy, thời gian tường {t:.1f} s\n")


if __name__ == "__main__":
    main()
