# -*- coding: utf-8 -*-
"""Kiểm thử PSO tìm trọng số thuộc tính và đối chứng tìm ngẫu nhiên. Chạy: python3 kiem_thu_pso_trong_so.py"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from cach_ket_hop import W_MAX, W_MIN, bai_toan_trong_so
from giao_dien import fcm_kpp, kiem_tra_giao_dien
from pso_trong_so import N_HAT, ngau_nhien_trong_so, pso_trong_so

NGAN_SACH_THU = 60


def _iris():
    X = load_iris().data.astype(np.float64)
    return (X - X.mean(axis=0)) / X.std(axis=0)


def test_dat_giao_dien():
    for fit in (pso_trong_so, ngau_nhien_trong_so):
        try:
            kiem_tra_giao_dien(fit, _iris(), 3, max_fe=NGAN_SACH_THU)
        except AssertionError as exc:
            raise AssertionError(f"{fit.__name__}: {exc}") from exc


def test_dung_het_ngan_sach_va_trong_so_trong_mien():
    for fit in (pso_trong_so, ngau_nhien_trong_so):
        kq = fit(_iris(), 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU)
        w = kq.thong_tin["trong_so"]
        assert kq.n_fe == NGAN_SACH_THU, (fit.__name__, kq.n_fe)
        assert w.shape == (4,) and np.all((w >= W_MIN) & (w <= W_MAX)), (fit.__name__, w)


def test_khong_te_hon_fcm_thuong_theo_fitness():
    """Cả 2 cách đều chấm w = 1 (FCM thường trên cùng V0), nên fitness tìm được không lớn hơn fitness của w = 1."""
    X = _iris()
    _, fitness = bai_toan_trong_so(X, 3, 2.0, np.random.default_rng(0))
    f_fcm = fitness(np.ones(X.shape[1]))
    for fit in (pso_trong_so, ngau_nhien_trong_so):
        kq = fit(X, 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU)
        assert kq.thong_tin["fitness"] <= f_fcm + 1e-12, (fit.__name__, kq.thong_tin["fitness"], f_fcm)
        assert np.isclose(kq.lich_su[-1], kq.thong_tin["fitness"]), fit.__name__


def test_lich_su_cung_nhip_hai_cach():
    """Lịch sử của cả 2 cách có một điểm sau mỗi N_HAT lần chấm, để vẽ chung một trục."""
    X = _iris()
    a = pso_trong_so(X, 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU)
    b = ngau_nhien_trong_so(X, 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU)
    assert len(a.lich_su) == len(b.lich_su) == NGAN_SACH_THU // N_HAT


def test_trong_so_deu_cho_lai_fcm_thuong():
    """w = 1 cho đúng phân hoạch FCM k-means++ cùng seed: đánh trọng số đều không đổi bài toán."""
    from cach_ket_hop import phan_hoach_trong_so
    X = _iris()
    V0, _ = bai_toan_trong_so(X, 3, 2.0, np.random.default_rng(0))
    U, V, _, _ = phan_hoach_trong_so(X, 2.0, V0, np.ones(X.shape[1]))
    kq = fcm_kpp(X, 3, 2.0, np.random.default_rng(0))
    assert np.allclose(U, kq.U, atol=1e-6) and np.allclose(V, kq.V, atol=1e-4)



def test_moi_phien_ban_dung_ngan_sach_va_khong_te_hon_fcm():
    from pso_khoi_tao import PHIEN_BAN
    X = _iris()
    _, fitness = bai_toan_trong_so(X, 3, 2.0, np.random.default_rng(0))
    f_fcm = fitness(np.ones(X.shape[1]))
    for pb in PHIEN_BAN:
        for n_hat in (5, 10):
            kq = pso_trong_so(X, 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU, phien_ban=pb, n_hat=n_hat)
            w = kq.thong_tin["trong_so"]
            assert kq.n_fe == NGAN_SACH_THU and kq.thong_tin["phien_ban"] == pb, (pb, n_hat, kq.n_fe)
            assert np.all((w >= W_MIN) & (w <= W_MAX)), (pb, w)
            assert kq.thong_tin["fitness"] <= f_fcm + 1e-12, (pb, n_hat)


def test_phien_ban_sai_bao_loi():
    try:
        pso_trong_so(_iris(), 3, 2.0, np.random.default_rng(0), NGAN_SACH_THU, phien_ban="khong_co")
    except ValueError:
        return
    raise AssertionError("phải báo ValueError khi phiên bản không hợp lệ")


def test_wfcm_giai_tich_tra_trong_so_dung_phan_hoach():
    """Trọng số WFCM lưu ra (λ = w^β) áp lại qua phan_hoach_trong_so từ cùng V0 cho cùng phân hoạch cứng (so bằng ARI
    vì số hiệu cụm có thể hoán vị)."""
    from sklearn.metrics import adjusted_rand_score

    from cach_ket_hop import phan_hoach_trong_so, wfcm_giai_tich
    from fcm import FCM
    X = _iris()
    kq = wfcm_giai_tich(X, 3, 2.0, np.random.default_rng(0))
    lam = kq.thong_tin["trong_so"]
    assert lam.shape == (4,) and np.all(lam > 0)
    V0 = FCM(3).init_centroids_kmeans_plusplus(X, seed=np.random.default_rng(0))
    U, _, _, _ = phan_hoach_trong_so(X, 2.0, V0, lam)
    assert adjusted_rand_score(U.argmax(1), kq.U.argmax(1)) > 0.99


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ĐẠT   {name}")
        except Exception as exc:
            failed += 1
            print(f"LỖI   {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} phép thử đạt")
    raise SystemExit(1 if failed else 0)
