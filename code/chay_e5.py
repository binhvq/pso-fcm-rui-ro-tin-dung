# -*- coding: utf-8 -*-
"""Thí nghiệm E5: PSO gốc và các phiên bản cải tiến làm bộ khởi tạo tâm cho FCM.

    python3 chay_e5.py [--nhanh] [--tien-trinh 8] [--ket-qua DIR]

Một tác vụ = (bộ, c, seed): chạy TUẦN TỰ 3 đối chứng (FCM U ngẫu nhiên, FCM từ c điểm ngẫu nhiên, FCM k-means++) và
mọi tổ hợp (phiên bản PSO × số hạt × số vòng), mỗi phương pháp nhận default_rng(seed) riêng. Một dòng mỗi
(phương pháp, seed) trong <ket-qua>/E5/<bo>_c<c>.csv; chạy lại cùng lệnh chỉ chạy phần thiếu. --nhanh: 2 seed.
"""

from __future__ import annotations

import os

for _bien in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_bien, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chay_thuc_nghiem import KET_QUA_MAC_DINH, M, SEED_DAU, SO_SEED, nap  # noqa: E402
from danh_gia import COT, danh_gia, mau_silhouette  # noqa: E402
from giao_dien import fcm_kpp, fcm_mot_lan  # noqa: E402
from pso_khoi_tao import PHIEN_BAN, fcm_diem_ngau_nhien, ham_e5  # noqa: E402

CAU_HINH = [("viet_nam", 3), ("taiwan_kiem_chung", 3), ("taiwan_kiem_chung_khong_log", 6)]
SO_HAT = (10, 20, 30)
SO_VONG = (20, 30, 50)
SO_SEED_NHANH = 2
DOI_CHUNG = {"fcm": fcm_mot_lan, "fcm_diem": fcm_diem_ngau_nhien, "fcm_kpp": fcm_kpp}
HAM = {**DOI_CHUNG, **{f"pso_{pb}_n{n}_t{t}": ham_e5(pb, n, t) for pb in PHIEN_BAN for n in SO_HAT for t in SO_VONG}}
COT_THEM = ["phien_ban", "n_hat", "so_vong", "j_pso"]


class TacVu(NamedTuple):
    bo: str
    c: int
    seed: int


def chay_tac_vu(tv: TacVu, phuong_phap: list[str] | None = None) -> list[dict]:
    bo = nap(tv.bo)
    mau = mau_silhouette(len(bo.X))
    dong = []
    for ten in phuong_phap or list(HAM):
        khoa = dict(thi_nghiem="E5", bo=tv.bo, c=tv.c, seed=tv.seed, phuong_phap=ten)
        try:
            kq = HAM[ten](bo.X, tv.c, M, np.random.default_rng(tv.seed), None)
            tt = kq.thong_tin
            them = dict(phien_ban=tt.get("phien_ban", ten), n_hat=tt.get("n_hat", 0), so_vong=tt.get("so_vong", 0),
                        j_pso=tt.get("j_pso", np.nan))
            dong.append({**khoa, **them, **danh_gia(bo.X, bo.y, kq, M, mau), "loi": ""})
        except Exception as exc:  # một phương pháp lỗi không dừng cả tác vụ; phân tích soát cột loi
            dong.append({**khoa, **{k: np.nan for k in COT_THEM + COT}, "loi": repr(exc)})
    return dong


def _tep(thu_muc: str, bo: str, c: int) -> str:
    return os.path.join(thu_muc, "E5", f"{bo}_c{c}.csv")


def _da_xong(thu_muc: str, tv: TacVu) -> bool:
    tep = _tep(thu_muc, tv.bo, tv.c)
    if not os.path.exists(tep):
        return False
    d = pd.read_csv(tep, usecols=["seed", "phuong_phap", "loi"])
    d = d[(d["seed"] == tv.seed) & d["loi"].fillna("").eq("")]
    return set(HAM) <= set(d["phuong_phap"])


def _ghi(thu_muc: str, tv: TacVu, dong: list[dict]) -> None:
    tep = _tep(thu_muc, tv.bo, tv.c)
    os.makedirs(os.path.dirname(tep), exist_ok=True)
    moi = pd.DataFrame(dong)
    if os.path.exists(tep):
        cu = pd.read_csv(tep)
        cu = cu[cu["seed"] != tv.seed]
        moi = pd.concat([cu, moi], ignore_index=True)
    moi.sort_values(["seed", "phuong_phap"]).to_csv(tep, index=False)


def _lam(tv: TacVu):
    t0 = time.perf_counter()
    return tv, chay_tac_vu(tv), time.perf_counter() - t0


def chay(thu_muc: str, nhanh: bool = False, so_tien_trinh: int = 8, nhat_ky=sys.stdout) -> int:
    so_seed = SO_SEED_NHANH if nhanh else SO_SEED
    viec = [tv for bo, c in CAU_HINH for s in range(so_seed)
            if not _da_xong(thu_muc, tv := TacVu(bo, c, SEED_DAU + s))]
    viec.sort(key=lambda tv: tv.bo != "viet_nam")  # Việt Nam nặng nhất (FCM hội tụ chậm) chạy trước
    pool = Pool(so_tien_trinh) if so_tien_trinh > 1 else None
    ket_qua = pool.imap_unordered(_lam, viec, chunksize=1) if pool else map(_lam, viec)
    try:
        for i, (tv, dong, t) in enumerate(ket_qua, 1):
            _ghi(thu_muc, tv, dong)
            loi = sum(1 for d in dong if d.get("loi"))
            print(f"[{i}/{len(viec)}] E5 {tv.bo} c={tv.c} {tv.seed}: {len(dong)} dòng, {t:.1f} s"
                  f"{f', {loi} LỖI' if loi else ''}", file=nhat_ky, flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return len(viec)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nhanh", action="store_true")
    ap.add_argument("--tien-trinh", type=int, default=8)
    ap.add_argument("--ket-qua", default=KET_QUA_MAC_DINH)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    n = chay(os.path.abspath(a.ket_qua), a.nhanh, a.tien_trinh)
    print(f"E5: {n} tác vụ, {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
