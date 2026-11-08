# -*- coding: utf-8 -*-
"""PSO tìm trọng số thuộc tính cho FCM (thí nghiệm E8) và đối chứng tìm ngẫu nhiên cùng ngân sách.

FCM coi mọi thuộc tính đã chuẩn hoá ngang nhau và không có cơ chế tự chọn trọng số theo chất lượng phân cụm; đây là
chỗ PSO tối ưu thứ FCM không tự tối ưu được (O'Neill và cộng sự 2018; Lu và cộng sự 2011). Mỗi hạt là vector trọng số
w ∈ [0,05; 1]^D; fitness là −Silhouette trên không gian gốc của phân hoạch FCM chạy trên dữ liệu đã nhân trọng số, cộng
phạt cụm nhỏ (cach_ket_hop.bai_toan_trong_so). Nhãn vỡ nợ không đi vào fitness, chỉ dùng để chấm kết quả.

    pso_trong_so          Mặc định PSO-W (ω 0,9 → 0,4, c1 = c2 = 2, như mục 2.3.1), N_HAT hạt, đủ NGAN_SACH lần chấm,
                          không dừng sớm; hạt đầu là w = 1 (FCM thường) nên kết quả không tệ hơn FCM theo fitness.
                          phien_ban chọn một trong 5 phiên bản của pso_khoi_tao (goc, w, s, log, qpso) với cùng tham số,
                          để so các phiên bản trong vai trò tìm trọng số (E9).
    ngau_nhien_trong_so   Rút NGAN_SACH − 1 vector w đều trong miền, cộng w = 1, giữ vector có fitness nhỏ nhất. Cùng
                          ngân sách, cùng V0 và cùng mẫu Silhouette với PSO khi cùng seed: phần chênh giữa 2 cách là
                          phần đóng góp của cơ chế bầy đàn, không phải của việc đánh trọng số.

Cả hai lưu trọng số tìm được vào thong_tin["trong_so"] để diễn giải thuộc tính nào quyết định phân khúc.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cach_ket_hop import W_MAX, W_MIN, bai_toan_trong_so, phan_hoach_trong_so
from giao_dien import KetQua
from pso_co_so import pso
from pso_khoi_tao import C_GOC, OMEGA, PHIEN_BAN
from thuat_toan_b import qpso

N_HAT = 10
NGAN_SACH = 500


def _ket_qua(X, m, V0, w, f, lich_su, n_fe, t0, t1, t2) -> KetQua:
    U, V, n_vong, t = phan_hoach_trong_so(X, m, V0, w)
    return KetQua(U=U, V=V, lich_su=lich_su, n_fe=n_fe, n_vong_fcm=n_vong, t_khoi_tao=t1 - t0,
                  t_tim_kiem=t2 - t1, t_tinh_chinh=t, thong_tin={"trong_so": np.asarray(w, float), "fitness": f})


def pso_trong_so(X, c, m, rng, max_fe=None, phien_ban: str = "w", n_hat: int = N_HAT) -> KetQua:
    if phien_ban not in PHIEN_BAN:
        raise ValueError(f"phiên bản PSO không hợp lệ: {phien_ban}")
    ngan_sach = NGAN_SACH if max_fe is None else max_fe
    t0 = perf_counter()
    d = X.shape[1]
    V0, fitness = bai_toan_trong_so(X, c, m, rng)
    bay = np.vstack([np.ones(d), rng.uniform(W_MIN, W_MAX, size=(n_hat - 1, d))])
    t1 = perf_counter()
    so_vong = ngan_sach // n_hat - 1
    if phien_ban == "qpso":
        gbest, lich_su, n_fe, _ = qpso(bay, W_MIN, W_MAX, fitness, rng, ngan_sach, max_vong=so_vong, dung_tri_tre=False)
        f = lich_su[-1]
    else:
        kq = pso(bay, W_MIN, W_MAX, fitness, rng, max_fe=ngan_sach, omega=OMEGA[phien_ban], max_vong=so_vong,
                 c1=C_GOC, c2=C_GOC, dung_tri_tre=False)
        gbest, f, lich_su, n_fe = kq.gbest, kq.gbest_f, kq.lich_su, kq.n_fe
    t2 = perf_counter()
    kq = _ket_qua(X, m, V0, gbest, f, lich_su, n_fe, t0, t1, t2)
    kq.thong_tin.update(phien_ban=phien_ban, n_hat=n_hat)
    return kq


def ngau_nhien_trong_so(X, c, m, rng, max_fe=None) -> KetQua:
    ngan_sach = NGAN_SACH if max_fe is None else max_fe
    t0 = perf_counter()
    d = X.shape[1]
    V0, fitness = bai_toan_trong_so(X, c, m, rng)
    ung_vien = np.vstack([np.ones(d), rng.uniform(W_MIN, W_MAX, size=(ngan_sach - 1, d))])
    t1 = perf_counter()
    f = np.array([fitness(w) for w in ung_vien])
    t2 = perf_counter()
    lich_su = np.minimum.accumulate(f)[N_HAT - 1::N_HAT].tolist()  # tốt nhất sau mỗi N_HAT lần chấm, để so với PSO
    i = int(np.argmin(f))
    return _ket_qua(X, m, V0, ung_vien[i], float(f[i]), lich_su, ngan_sach, t0, t1, t2)
