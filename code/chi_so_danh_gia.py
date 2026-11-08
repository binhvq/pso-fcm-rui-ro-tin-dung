# -*- coding: utf-8 -*-
"""Chỉ số đánh giá phân cụm, cài bằng NumPy.

Chỉ số nội (không cần nhãn): PC, PE, XB, DB mờ.
Chỉ số ngoại (so với nhãn thật, ở đây là nhãn vỡ nợ): ARI, NMI, ACC.
Dấu + / - sau tên: cao hơn tốt hơn / thấp hơn tốt hơn.
"""

from __future__ import annotations

import numpy as np

from fcm import cdist_euclidean, cdist_sq_euclidean


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Bảng tương liên giữa 2 phân hoạch."""
    t = np.unique(y_true, return_inverse=True)[1]
    p = np.unique(y_pred, return_inverse=True)[1]
    cm = np.zeros((t.max() + 1, p.max() + 1), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def pc(membership: np.ndarray) -> float:
    """PC+ (Partition Coefficient) = (1/N) Σᵢ Σₖ uᵢₖ² ∈ [1/C, 1].

    Ref: Bezdek J.C. (1981). Pattern Recognition with Fuzzy Objective Function
         Algorithms. Plenum Press.
    """
    return float(np.mean(np.sum(membership**2, axis=1)))


def pe(membership: np.ndarray, eps: float = 1e-9) -> float:
    """PE- (Normalized Partition Entropy) ∈ [0, 1].

    PE = -(1/N) Σᵢ Σₖ uᵢₖ log₂ uᵢₖ / log₂ C.
    """
    C = membership.shape[1]
    safe = np.maximum(membership, eps)
    entropy = -np.sum(safe * np.log2(safe), axis=1)
    return float(np.mean(entropy) / np.log2(C))


def xb(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray, m: float = 2.0) -> float:
    """XB- (Xie–Beni) = Σᵢ Σₖ uᵢₖᵐ ||xᵢ - vₖ||² / (N · min_{k≠l} ||vₖ - vₗ||²).

    Ref: Xie X.L., Beni G. (1991). IEEE TPAMI 13(8), 841-847.
    """
    if centroids.shape[0] <= 1:
        return float("inf")
    compactness = float(np.sum((membership**m) * cdist_sq_euclidean(data, centroids)))
    v_dist_sq = cdist_sq_euclidean(centroids, centroids)
    np.fill_diagonal(v_dist_sq, np.inf)
    min_dist_sq = float(np.min(v_dist_sq))
    if min_dist_sq <= 0:  # 2 tâm trùng nhau: tách biệt bằng 0 → XB vô cùng
        return float("inf")
    return compactness / (len(data) * min_dist_sq)


def db(data: np.ndarray, centroids: np.ndarray, membership: np.ndarray, m: float = 2.0, eps: float = 1e-9) -> float:
    """DB- (Davies–Bouldin bản mờ) = (1/C) Σᵢ max_{j≠i} (Sᵢ + Sⱼ) / d(vᵢ, vⱼ),
    với Sᵢ = √[Σₖ uₖᵢᵐ ||xₖ - vᵢ||² / Σₖ uₖᵢᵐ].

    Ref: Davies D.L., Bouldin D.W. (1979). IEEE TPAMI 1(2), 224-227.
    """
    u_m = membership**m
    numerator = np.sum(u_m * cdist_sq_euclidean(data, centroids), axis=0)
    weight = np.sum(u_m, axis=0)
    S = np.sqrt(numerator / np.where(weight == 0, eps, weight))
    centroid_dists = cdist_euclidean(centroids, centroids)
    np.fill_diagonal(centroid_dists, np.inf)
    with np.errstate(divide="ignore"):  # 2 tâm trùng nhau → R vô cùng, đúng định nghĩa
        R = (S[:, None] + S[None, :]) / centroid_dists
    return float(np.mean(np.max(R, axis=1)))


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ARI+ (Adjusted Rand Index) ∈ [-1, 1].

    Ref: Hubert L., Arabie P. (1985). J. Classification 2(1), 193-218.
    """
    cm = _contingency(y_true, y_pred).astype(np.float64)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(cm).sum()
    sum_i = comb2(cm.sum(axis=1)).sum()
    sum_j = comb2(cm.sum(axis=0)).sum()
    expected = sum_i * sum_j / comb2(cm.sum())
    max_index = 0.5 * (sum_i + sum_j)
    if max_index == expected:
        return 0.0
    return float((sum_ij - expected) / (max_index - expected))


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NMI+ (arithmetic) = 2·I(Y; Ŷ) / [H(Y) + H(Ŷ)] ∈ [0, 1].

    Ref: Strehl A., Ghosh J. (2002). JMLR 3, 583-617.
    """
    cm = _contingency(y_true, y_pred).astype(np.float64)
    pij = cm / cm.sum()
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    nz = pij > 0
    mutual_info = float(np.sum(pij[nz] * np.log(pij[nz] / (pi @ pj)[nz])))
    h_true = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
    h_pred = -float(np.sum(pj[pj > 0] * np.log(pj[pj > 0])))
    denom = 0.5 * (h_true + h_pred)
    return 0.0 if denom <= 0 else mutual_info / denom


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ACC+ không ánh xạ nhãn."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def acc_optimize(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ACC+ với ánh xạ cụm → nhãn tối ưu (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    cm = _contingency(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return float(cm[row_ind, col_ind].sum() / cm.sum())
