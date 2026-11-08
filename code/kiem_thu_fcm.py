# -*- coding: utf-8 -*-
"""Kiểm thử FCM và chỉ số đánh giá. Chạy: python3 kiem_thu_fcm.py"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import chi_so_danh_gia as cs
from fcm import FCM


def _iris():
    data = load_iris()
    return data.data.astype(np.float64), data.target


def test_rang_buoc_do_thuoc():
    X, _ = _iris()
    U, V, _ = FCM(3).fit(X, seed=42)
    assert U.shape == (150, 3) and V.shape == (3, 4)
    assert np.all(U >= 0) and np.all(U <= 1)
    assert np.allclose(U.sum(axis=1), 1.0)


def test_ham_muc_tieu_khong_tang():
    X, _ = _iris()
    model = FCM(3)
    model._track_objective = True
    model.fit(X, seed=1)
    J = np.array(model.objective_history)
    assert np.all(np.diff(J) <= 1e-9 * J[:-1]), "J_m phải không tăng qua các vòng"


def test_khop_skfuzzy():
    import skfuzzy

    X, _ = _iris()
    model = FCM(3, epsilon=1e-8, max_iter=1000)
    model.fit(X, seed=0)
    V_ref, *_ = skfuzzy.cmeans(X.T, c=3, m=2.0, error=1e-8, maxiter=1000, seed=0)
    ours = np.sort(model.centroids[:, 0])
    ref = np.sort(V_ref[:, 0])
    assert np.allclose(ours, ref, atol=1e-4), (ours, ref)


def test_tai_lap_theo_seed():
    X, _ = _iris()
    U1, V1, _ = FCM(3).fit(X, seed=7)
    U2, V2, _ = FCM(3).fit(X, seed=7)
    assert np.array_equal(U1, U2) and np.array_equal(V1, V2)


def test_init_v_hoi_tu_theo_tam():
    X, _ = _iris()
    model = FCM(3, epsilon=1e-8)
    V0 = model.init_centroids_kmeans_plusplus(X, seed=3)
    U, V, _ = model.fit(X, init_v=V0, with_u=False)
    assert np.allclose(U, model.predict_proba(X), atol=1e-6)
    assert np.array_equal(model.predict(X), model.labels)


def test_float32_va_chebyshev():
    X, _ = _iris()
    for kw in ({"dtype": "float32"}, {"metric": "chebyshev"}, {"m": 1.1}):
        U, V, _ = FCM(3, **kw).fit(X, seed=2)
        assert np.all(np.isfinite(U)) and np.allclose(U.sum(axis=1), 1.0, atol=1e-5), kw


def test_tham_so_sai():
    for kw in ({"n_clusters": 3, "m": 1.0}, {"n_clusters": 0}, {"n_clusters": 3, "metric": "cosine"}):
        try:
            FCM(**kw)
        except ValueError:
            continue
        raise AssertionError(f"phải báo lỗi với {kw}")


def test_bao_loi_khi_co_nan():
    X, _ = _iris()
    X[5, 2] = np.nan
    try:
        FCM(3).fit(X, seed=0)
    except ValueError:
        return
    raise AssertionError("phải báo lỗi khi dữ liệu có NaN")


def test_khong_dong_rng_toan_cuc():
    X, _ = _iris()
    np.random.seed(123)
    expected = np.random.rand(3)
    np.random.seed(123)
    FCM(3).fit(X, seed=0)
    FCM(3).init_centroids_kmeans_plusplus(X, seed=0)
    assert np.array_equal(np.random.rand(3), expected)


def test_init_v_khong_tieu_thu_so_ngau_nhien():
    X, _ = _iris()
    rng = np.random.default_rng(5)
    V0 = X[[0, 50, 100]]
    FCM(3).fit(X, init_v=V0, with_u=False, seed=rng)
    assert rng.random() == np.random.default_rng(5).random()


def test_generator_dung_chung_tai_lap():
    X, _ = _iris()
    U1, _, _ = FCM(3).fit(X, seed=np.random.default_rng(9))
    U2, _, _ = FCM(3).fit(X, seed=9)
    assert np.array_equal(U1, U2)


def test_objective_from_centroids():
    X, _ = _iris()
    model = FCM(3)
    V = X[[0, 50, 100]]
    U = model._update_membership(X, V)
    assert np.isclose(model.objective_from_centroids(X, V), model.compute_objective_j(X, U, V))
    U_fit, V_fit, _ = model.fit(X, init_v=V, with_u=False, seed=0)
    assert model.objective_from_centroids(X, V_fit) <= model.objective_from_centroids(X, V) + 1e-9


def test_chi_so_khop_sklearn():
    X, y = _iris()
    model = FCM(3)
    U, V, _ = model.fit(X, seed=42)
    labels = model.labels
    assert np.isclose(cs.ari(y, labels), adjusted_rand_score(y, labels))
    assert np.isclose(cs.nmi(y, labels), normalized_mutual_info_score(y, labels))
    U_hard = np.eye(3)[labels]
    assert 1 / 3 <= cs.pc(U) <= 1 and 0 <= cs.pe(U) <= 1
    assert np.isclose(cs.pc(U_hard), 1.0) and cs.pe(U_hard) < 1e-6
    assert cs.xb(X, V, U) > 0 and cs.db(X, V, U) > 0 and cs.acc_optimize(y, labels) > 0.85


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
