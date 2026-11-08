# -*- coding: utf-8 -*-
"""Thuật toán C: MSGWO–WFCM (Liu, Wang & Chao 2021, J. Electrical and Computer Engineering, 7387153).

Sói là bộ tâm c×D; mỗi sói giữ ma trận trọng số đặc trưng W (c×D, theo từng cụm) tính giải tích, không do GWO tìm.
Chấm một sói (thứ tự theo Algorithm 3): U ← Eq. 5 (với W cũ), W ← Eq. 7, J_WFCM ← Eq. 1; fitness 1/J (Eq. 15), tức
cực tiểu J_WFCM. Ba chiến lược của MSGWO:
    1. khởi tạo nửa ngẫu nhiên, nửa chia đoạn đều (Algorithm 1);
    2. hệ số hội tụ a(t) = 2 − 2·(t/t_max)^k với k = 1 cho phân cụm (Bảng 5) — trùng lịch tuyến tính của GWO gốc;
    3. học đối lập tổng quát (GOBL) mỗi vòng cho cả quần thể, giữ n cá thể tốt nhất của quần thể ∪ quần thể đối.
       Bài không cho công thức GOBL; dùng dạng của Wang và cộng sự (2011): x* = r·(da + db) − x với da, db là biên
       động của quần thể, r ~ U(0,1) một giá trị mỗi vòng; ngoài biên thì rút đều trong [da, db]. [GIẢ ĐỊNH]
Đầu ra là phân hoạch WFCM của sói tốt nhất từng gặp (không chạy FCM thêm).

Khác bài (ghi trong Ch.2): m = 2 (bài 1,5), 20 sói (bài 50), t_max = 150 và dừng trì trệ chung (bài không nêu t_max, ξ);
mũ trọng số t = 1,5 giữ như bài; W khởi tạo 1/D; biên tìm kiếm [min X, max X].
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from giao_dien import KetQua
from pso_co_so import MAX_VONG, N_HAT, STALL_ITERS, STALL_TOL

MU_TRONG_SO = 1.5
K_PHI_TUYEN = 1.0
EPS = 1e-10


def _wfcm_cham(X, V, W, m, t=MU_TRONG_SO):
    """Một lần chấm theo Algorithm 3: U từ (V, W cũ), W mới từ (U, V), rồi J_WFCM(U, V, W mới)."""
    diff2 = (X[:, None, :] - V[None, :, :]) ** 2  # (n, c, D)
    d = np.einsum("ncd,cd->nc", diff2, W**t) + EPS
    nghich = d ** (-1.0 / (m - 1.0))
    U = nghich / nghich.sum(axis=1, keepdims=True)  # Eq. 5
    S = np.einsum("nc,ncd->cd", U**m, diff2) + EPS  # Σ_i u_ci^m (x_id − v_cd)²
    W_moi = S ** (-1.0 / (t - 1.0))
    W_moi /= W_moi.sum(axis=1, keepdims=True)  # Eq. 7
    J = float(np.sum((U**m) * np.einsum("ncd,cd->nc", diff2, W_moi**t)))  # Eq. 1
    return J, U, W_moi


def _khoi_tao_da_dang(n, lb, ub, rng):
    """Algorithm 1: nửa quần thể ngẫu nhiên đều; nửa còn lại mỗi chiều chia n/2 đoạn bằng nhau, mỗi đoạn dùng một lần."""
    nua = n // 2
    ngau_nhien = rng.uniform(lb, ub, size=(nua,) + lb.shape)
    k = n - nua
    doan = np.stack([rng.permutation(k) for _ in range(lb.size)], axis=-1).reshape((k,) + lb.shape)
    deu = lb + (doan + rng.random((k,) + lb.shape)) * (ub - lb) / k
    return np.concatenate([ngau_nhien, deu])


def msgwo_wfcm(X, c, m, rng, max_fe=None, n_soi: int = N_HAT, t_max: int = MAX_VONG) -> KetQua:
    t0 = perf_counter()
    D = X.shape[1]
    lb = np.broadcast_to(X.min(axis=0), (c, D)).copy()
    ub = np.broadcast_to(X.max(axis=0), (c, D)).copy()
    soi = _khoi_tao_da_dang(n_soi, lb, ub, rng)
    W = np.full((n_soi, c, D), 1.0 / D)
    J = np.empty(n_soi)
    for i in range(n_soi):
        J[i], _, W[i] = _wfcm_cham(X, soi[i], W[i], m)
    n_fe = n_soi
    t1 = perf_counter()
    i0 = int(np.argmin(J))
    tot = (float(J[i0]), soi[i0].copy(), W[i0].copy())  # nghiệm tốt nhất từng gặp (GWO di chuyển cả sói α)
    lich_su, n_vong, tri_tre = [tot[0]], 0, 0
    for it in range(1, t_max + 1):
        if max_fe is not None and n_fe + 2 * n_soi > max_fe:
            break
        thu_tu = np.argsort(J)
        dau_dan = soi[thu_tu[:3]]  # α, β, δ: J nhỏ nhất
        a = 2.0 - 2.0 * (it / t_max) ** K_PHI_TUYEN  # Eq. 14
        moi = np.zeros_like(soi)
        for l in range(3):
            A = 2.0 * a * rng.random(soi.shape) - a  # Eq. 9
            C = 2.0 * rng.random(soi.shape)  # Eq. 10
            moi += dau_dan[l] - A * np.abs(C * dau_dan[l] - soi)  # Eq. 11–12
        soi = np.clip(moi / 3.0, lb, ub)  # Eq. 13
        for i in range(n_soi):
            J[i], _, W[i] = _wfcm_cham(X, soi[i], W[i], m)
        # GOBL [GIẢ ĐỊNH theo Wang và cộng sự 2011]
        da, db = soi.min(axis=0), soi.max(axis=0)
        doi = rng.random() * (da + db) - soi
        ngoai = (doi < lb) | (doi > ub)
        doi[ngoai] = rng.uniform(np.broadcast_to(da, doi.shape)[ngoai], np.broadcast_to(db, doi.shape)[ngoai])
        W_doi = np.full_like(W, 1.0 / D)
        J_doi = np.empty(n_soi)
        for i in range(n_soi):
            J_doi[i], _, W_doi[i] = _wfcm_cham(X, doi[i], W_doi[i], m)
        n_fe += 2 * n_soi
        gop = np.argsort(np.concatenate([J, J_doi]))[:n_soi]
        soi = np.concatenate([soi, doi])[gop]
        W = np.concatenate([W, W_doi])[gop]
        J = np.concatenate([J, J_doi])[gop]
        n_vong = it
        truoc = lich_su[-1]
        if J[0] < tot[0]:
            tot = (float(J[0]), soi[0].copy(), W[0].copy())
        lich_su.append(tot[0])
        tri_tre = tri_tre + 1 if (truoc - lich_su[-1]) <= STALL_TOL * abs(truoc) else 0
        if tri_tre >= STALL_ITERS:
            break
    t2 = perf_counter()
    J_tot, V_tot, W_tot = tot
    _, U, _ = _wfcm_cham(X, V_tot, W_tot, m)
    return KetQua(U=U, V=V_tot, lich_su=lich_su, n_fe=n_fe, t_khoi_tao=t1 - t0, t_tim_kiem=t2 - t1,
                  t_tinh_chinh=perf_counter() - t2, thong_tin=dict(n_vong=n_vong, J_wfcm=J_tot))
