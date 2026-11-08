# -*- coding: utf-8 -*-
"""Fuzzy C-Means (Bezdek, Ehrlich & Full, 1984) cài bằng NumPy thuần.

Chuyển từ bản CuPy/NumPy của thư viện `ds` (v2.0-opt): giữ nguyên thuật toán,
API công khai và các điểm override để PSO–FCM và các biến thể kế thừa; bỏ toàn
bộ phần chọn thiết bị GPU, kernel CUDA và đồng bộ trễ GPU→CPU. Khác bản gốc:
dùng np.random.Generator riêng thay cho RNG toàn cục, báo lỗi khi dữ liệu có
NaN, đo thời gian bằng perf_counter.

Bài toán: chia X = {x_1..x_n} ⊂ R^d thành C cụm mờ với U ∈ [0,1]^{n×C},
Σ_k u_ik = 1, tối thiểu

    J_m(U, V) = Σ_i Σ_k u_ik^m · ||x_i - v_k||²            (m > 1)

Cập nhật luân phiên đến khi max|U^(t+1) - U^(t)| < ε hoặc chạm max_iter:

    v_k  = Σ_i u_ik^m x_i / Σ_i u_ik^m
    u_ik = d_ik^(-2/(m-1)) / Σ_l d_il^(-2/(m-1))

Ref: Bezdek J.C., Ehrlich R., Full W. (1984). FCM: The fuzzy c-means clustering
     algorithm. Computers & Geosciences 10(2-3), 191-203.
     DOI 10.1016/0098-3004(84)90020-7
"""

from __future__ import annotations

import time

import numpy as np


class ClusteringDefaults:
    """Tham số mặc định dùng chung cho các thuật toán phân cụm."""

    M: float = 2.0  # Hệ số mờ (fuzzifier), m > 1
    EPSILON: float = 1e-5  # Ngưỡng hội tụ trên max|ΔU|
    MAX_ITER: int = 10000
    SEED: int = 42
    METRIC: str = "euclidean"


CD = ClusteringDefaults


def as_rng(seed=None) -> np.random.Generator:
    """Chuẩn hoá seed (int | Generator | None) thành Generator riêng.

    Không đụng tới RNG toàn cục np.random, nên PSO và FCM dùng chung một
    Generator mà không làm lệch dãy số của nhau.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def cdist_sq_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ma trận bình phương khoảng cách Euclid (N, M) giữa A (N, D) và B (M, D)."""
    a_sq = np.sum(A * A, axis=1, keepdims=True)
    b_sq = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(a_sq - 2.0 * (A @ B.T) + b_sq, 0.0)


def cdist_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ma trận khoảng cách Euclid (N, M)."""
    return np.sqrt(cdist_sq_euclidean(A, B))


class FCM:
    """
    Fuzzy C-Means Clustering (Bezdek 1984).

    Parameters
    ----------
    n_clusters : int
        Số cụm C (>= 1).
    m : float, default=2
        Hệ số mờ. m → 1 thoái về k-means cứng, m → ∞ độ thuộc tiến tới đều.
    epsilon : float, default=1e-5
        Dừng khi max|U^(t+1) - U^(t)| < ε.
    max_iter : int, default=10000
    metric : {'euclidean', 'chebyshev'}, default='euclidean'
    dtype : {'float64', 'float32'}, default='float64'
    index : int, default=0
        ID phân biệt nhiều instance trong cùng thí nghiệm.

    Attributes
    ----------
    membership : (n, C) — ma trận U sau fit.
    centroids : (C, d) — tâm cụm V sau fit.
    labels : (n,) — argmax(U).
    n_iter, process_time, objective_history (khi _track_objective=True).

    Override points (cho lớp con)
    -----------------------------
    - Mức 1: _compute_squared_distances / _compute_distances / _compute_cost_matrix
    - Mức 2: _cost_to_membership
    - Mức 3: _update_membership
    - Mức 4: _compute_weights / _update_centroids
    - Mức 5: _on_fit_start / _on_iteration_end / _on_fit_end

    Examples
    --------
    >>> fcm = FCM(n_clusters=3, m=2)
    >>> U, V, n_iter = fcm.fit(X, seed=42)
    >>> y_pred = fcm.predict(X_test)
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = CD.M,
        epsilon: float = CD.EPSILON,
        max_iter: int = CD.MAX_ITER,
        metric: str = CD.METRIC,
        dtype: str = "float64",
        index: int = 0,
    ):
        if m <= 1:
            raise ValueError(f"Fuzzifier m must be > 1, got m={m}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if metric not in ("euclidean", "chebyshev"):
            raise ValueError(f"Unsupported metric: {metric}")
        if dtype not in ("float64", "float32"):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype}")

        self._n_clusters = n_clusters
        self._m = m
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._metric = metric
        self._dtype = np.dtype(dtype).type
        self._index = index

        self._X: np.ndarray | None = None
        self._U: np.ndarray | None = None
        self._V: np.ndarray | None = None
        self._x_sq_cache: np.ndarray | None = None

        self._n_iter = 0
        self._process_time = 0.0
        self._converged = False
        self._objective_history: list[float] = []

        # Ghi J mỗi vòng tốn thêm một lần tính ma trận chi phí, nên mặc định tắt.
        self._track_objective = False
        # Dừng sớm tuỳ chọn theo ΔJ tương đối.
        self._early_stop_on_j = False
        self._j_tol = 1e-5

    # =========================================================================
    # PUBLIC PROPERTIES
    # =========================================================================

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def m(self) -> float:
        return self._m

    @property
    def n_iter(self) -> int:
        return self._n_iter

    @property
    def process_time(self) -> float:
        return self._process_time

    @process_time.setter
    def process_time(self, value: float):
        self._process_time = value

    @property
    def membership(self) -> np.ndarray | None:
        return self._U

    @membership.setter
    def membership(self, value):
        self._U = None if value is None else self._as_array(value)

    @property
    def centroids(self) -> np.ndarray | None:
        return self._V

    @centroids.setter
    def centroids(self, value):
        self._V = None if value is None else self._as_array(value)

    @property
    def labels(self) -> np.ndarray | None:
        if self._U is None:
            return None
        return np.argmax(self._U, axis=1)

    def get_labels(self) -> np.ndarray | None:
        return self.labels

    @property
    def step(self) -> int:
        return self._n_iter

    @step.setter
    def step(self, value: int):
        self._n_iter = value

    @property
    def objective_history(self) -> list[float]:
        return self._objective_history

    @objective_history.setter
    def objective_history(self, value: list[float]):
        self._objective_history = value

    @property
    def local_data(self) -> np.ndarray | None:
        return self._X

    @local_data.setter
    def local_data(self, value):
        self._X = value

    @property
    def exited(self) -> bool:
        return self._converged

    @exited.setter
    def exited(self, value: bool):
        self._converged = value

    @property
    def index(self) -> int:
        return self._index

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def version(self) -> str:
        return "2.0-numpy"

    def _as_array(self, arr) -> np.ndarray:
        return np.asarray(arr, dtype=self._dtype)

    # =========================================================================
    # MỨC 1: KHOẢNG CÁCH / CHI PHÍ
    # =========================================================================

    def _compute_squared_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """||x_i - v_k||², dùng lại sum(X²) đã cache khi X là dữ liệu huấn luyện."""
        if X is self._X and self._x_sq_cache is not None:
            sum_x_sq = self._x_sq_cache
        else:
            sum_x_sq = np.sum(X * X, axis=1, keepdims=True)
        sum_v_sq = np.sum(V * V, axis=1, keepdims=True).T
        return np.maximum(sum_x_sq - 2 * (X @ V.T) + sum_v_sq, 0)

    def _compute_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        if self._metric == "euclidean":
            return np.sqrt(self._compute_squared_distances(X, V))
        # chebyshev: lặp theo cụm để bộ nhớ đỉnh là (N, D) thay vì (N, C, D)
        out = np.empty((X.shape[0], self._n_clusters), dtype=self._dtype)
        for k in range(self._n_clusters):
            out[:, k] = np.max(np.abs(X - V[k]), axis=1)
        return out

    def _compute_cost_matrix(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Chi phí = bình phương khoảng cách theo metric đã chọn."""
        if self._metric == "euclidean":
            return self._compute_squared_distances(X, V)
        d = self._compute_distances(X, V)
        return d * d

    def _distance_cdist(self, X, V):
        return self._compute_distances(X, V)

    def avoid_zero(self, arr, epsilon=None):
        if epsilon is None:
            epsilon = np.finfo(self._dtype).eps
        return np.fmax(arr, epsilon)

    def _clamp_array_to_epsilon(self, arr):
        eps = np.finfo(self._dtype).eps
        return np.where(arr < eps, eps, arr)

    # =========================================================================
    # MỨC 2: CHI PHÍ → ĐỘ THUỘC
    # =========================================================================

    def _cost_to_membership(self, costs: np.ndarray) -> np.ndarray:
        """u_ik ∝ cost_ik^(-1/(m-1)).

        Với float32 hoặc m gần 1, chia mỗi hàng cho giá trị nhỏ nhất trước khi
        luỹ thừa để tránh tràn số; phép chia bất biến theo tỉ lệ nên U không đổi.
        """
        eps = np.finfo(self._dtype).eps
        costs = np.fmax(costs, eps)
        if self._dtype == np.float32 or self._m < 1.3:
            row_min = np.fmax(costs.min(axis=1, keepdims=True), eps)
            costs = costs / row_min
        inv_costs = costs ** (-1.0 / (self._m - 1.0))
        return inv_costs / np.sum(inv_costs, axis=1, keepdims=True)

    def calculate_membership_by_distances(self, distances, m: float = None, squared: bool = False):
        if m is None:
            m = self._m
        costs = distances if squared else distances**2
        costs = np.fmax(costs, np.finfo(self._dtype).eps)
        inv_costs = costs ** (-1.0 / (m - 1.0))
        return inv_costs / np.sum(inv_costs, axis=1, keepdims=True)

    # =========================================================================
    # MỨC 3: CẬP NHẬT ĐỘ THUỘC
    # =========================================================================

    def _update_membership(self, X=None, V=None, U_old=None, *, data=None, centroids=None):
        if X is None:
            X = data
        if V is None:
            V = centroids
        return self._cost_to_membership(self._compute_cost_matrix(X, V))

    # =========================================================================
    # MỨC 4: CẬP NHẬT TÂM CỤM
    # =========================================================================

    def _compute_weights(self, U: np.ndarray) -> np.ndarray:
        return U**self._m

    def _update_centroids(self, X=None, U=None, V_old=None, *, data=None, membership=None):
        if X is None:
            X = data
        if U is None:
            U = membership
        weights = self._compute_weights(U)
        numerator = weights.T @ X
        denominator = np.sum(weights, axis=0, keepdims=True).T
        return numerator / np.fmax(denominator, np.finfo(self._dtype).eps)

    # =========================================================================
    # HỘI TỤ
    # =========================================================================

    def _check_convergence(self, old: np.ndarray, new: np.ndarray) -> bool:
        return float(np.abs(old - new).max()) < self._epsilon

    def check_exit_by_error(self, old, new) -> bool:
        self._converged = self._check_convergence(old, new)
        return self._converged

    # =========================================================================
    # MỨC 5: HOOKS
    # =========================================================================

    def _on_fit_start(self, X: np.ndarray):
        self._x_sq_cache = np.sum(X * X, axis=1, keepdims=True)

    def _on_iteration_end(self, step: int, U: np.ndarray, V: np.ndarray):
        pass

    def _on_fit_end(self, X, U, V):
        pass

    # =========================================================================
    # KHỞI TẠO
    # =========================================================================
    # seed nhận int | np.random.Generator | None (xem as_rng).

    def _init_centroids(self, X: np.ndarray, seed=None) -> np.ndarray:
        indices = as_rng(seed).choice(len(X), self._n_clusters, replace=False)
        return X[indices].copy()

    def _init_membership(self, n_samples: int, seed=None) -> np.ndarray:
        U = as_rng(seed).random((n_samples, self._n_clusters)).astype(self._dtype)
        return U / np.sum(U, axis=1, keepdims=True)

    def _init_centroid_random(self, data, seed=None):
        return self._init_centroids(self._as_array(data), seed)

    def _init_membership_random(self, n_samples, seed=None):
        return self._init_membership(n_samples, seed)

    def init_centroids_kmeans_plusplus(self, data, init_v=None, seed=None):
        rng = as_rng(seed)
        n_samples, n_features = data.shape
        centroids = np.empty((self._n_clusters, n_features), dtype=data.dtype)
        if init_v is None:
            init_v = data[rng.choice(n_samples)]
        centroids[0] = init_v
        for i in range(1, self._n_clusters):
            distances = np.min(cdist_euclidean(data, centroids[:i]), axis=1)
            probabilities = distances**2
            sum_probs = probabilities.sum()
            if sum_probs > 0:
                next_idx = rng.choice(n_samples, p=probabilities / sum_probs)
            else:
                next_idx = rng.choice(n_samples)
            centroids[i] = data[next_idx]
        return centroids

    def init_centroids_pca(self, data):
        from sklearn.decomposition import PCA

        projection_1d = PCA(n_components=1).fit_transform(data).flatten()
        split_indices = np.array_split(np.argsort(projection_1d), self._n_clusters)
        return np.array([np.mean(data[idx], axis=0) for idx in split_indices])

    def init_centroids_naive_sharding(self, data):
        split_indices = np.array_split(np.argsort(np.sum(data, axis=1)), self._n_clusters)
        return np.array([np.mean(data[idx], axis=0) for idx in split_indices])

    # =========================================================================
    # HÀM MỤC TIÊU
    # =========================================================================

    def _compute_objective_from_state(self, U: np.ndarray, V: np.ndarray) -> float:
        costs = self._compute_cost_matrix(self._X, V)
        return float(np.sum((U**self._m) * costs))

    def compute_objective(self) -> float:
        if self._X is None or self._U is None or self._V is None:
            raise RuntimeError("Model not fitted.")
        return self._compute_objective_from_state(self._U, self._V)

    def compute_objective_j(self, data, membership, centroids) -> float:
        X, U, V = self._as_array(data), self._as_array(membership), self._as_array(centroids)
        return float(np.sum((U**self._m) * self._compute_squared_distances(X, V)))

    def objective_from_centroids(self, data, centroids) -> float:
        """J_m(U*(V), V): tính U tối ưu theo V rồi tính J — hàm fitness cho PSO.

        Mã hoá cá thể là ma trận tâm (C, D), nên mỗi lần đánh giá fitness là
        một lần gọi hàm này.
        """
        X, V = self._as_array(data), self._as_array(centroids)
        costs = self._compute_cost_matrix(X, V)
        U = self._cost_to_membership(costs)
        return float(np.sum((U**self._m) * costs))

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(self, data, init_u=None, init_v=None, seed=None, with_u=True) -> tuple:
        """Huấn luyện FCM.

        with_u=True: hội tụ theo U (khởi tạo U, cập nhật V trước).
        with_u=False: hội tụ theo V (khởi tạo V, cập nhật U trước) — dùng khi
        PSO cung cấp tâm cụm ban đầu qua init_v.
        seed: int | np.random.Generator | None. Chỉ khởi tạo ngẫu nhiên thứ
        vòng lặp cần (U khi with_u=True, V khi with_u=False), nên truyền
        init_v thì không tiêu thụ số ngẫu nhiên nào.
        Trả về (U, V, số vòng lặp).
        """
        self._converged = False
        self._objective_history = []
        self._x_sq_cache = None

        self._X = self._as_array(data)
        if not np.all(np.isfinite(self._X)):
            raise ValueError("Dữ liệu có NaN hoặc vô cực — xử lý giá trị thiếu trước khi phân cụm.")
        if len(self._X) < self._n_clusters:
            raise ValueError(f"Số mẫu ({len(self._X)}) nhỏ hơn số cụm ({self._n_clusters}).")
        n_samples = len(self._X)
        rng = as_rng(seed)

        V = self._as_array(init_v) if init_v is not None else None
        U = self._as_array(init_u) if init_u is not None else None
        if with_u and U is None:
            U = self._init_membership(n_samples, rng)
        if not with_u and V is None:
            V = self._init_centroids(self._X, rng)

        self._on_fit_start(self._X)

        start_time = time.perf_counter()
        step = 0
        track_j = self._track_objective or self._early_stop_on_j

        for step in range(self._max_iter):
            if with_u:
                U_old = U.copy()
                V = self._update_centroids(self._X, U, V)
                U = self._update_membership(self._X, V, U)
                converged = self._check_convergence(U_old, U)
            else:
                V_old = V.copy()
                U = self._update_membership(self._X, V, U)
                V = self._update_centroids(self._X, U, V)
                converged = self._check_convergence(V_old, V)

            if track_j:
                self._objective_history.append(self._compute_objective_from_state(U, V))

            if self._early_stop_on_j and not converged and len(self._objective_history) >= 2:
                j_old, j_new = self._objective_history[-2], self._objective_history[-1]
                if abs(j_old - j_new) / max(abs(j_old), 1.0) < self._j_tol:
                    converged = True

            # Gọi hook trước khi thoát để vòng cuối vẫn quan sát được.
            self._on_iteration_end(step, U, V)

            if converged:
                self._converged = True
                break

        self._U = U
        self._V = V
        self._n_iter = step + 1
        self._process_time = time.perf_counter() - start_time

        self._on_fit_end(self._X, U, V)

        return self._U, self._V, self._n_iter

    # =========================================================================
    # DỰ ĐOÁN
    # =========================================================================

    def predict_proba(self, new_data) -> np.ndarray:
        if self._V is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._update_membership(self._as_array(new_data), self._V)

    def predict(self, new_data) -> np.ndarray:
        return np.argmax(self.predict_proba(new_data), axis=1)

    def get_clusters(self, labels=None) -> list:
        if labels is None:
            labels = self.labels
        return [self._X[labels == k] for k in range(self._n_clusters)]
