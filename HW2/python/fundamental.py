"""Step 2 of the spec: estimate the fundamental matrix using the
normalized 8-point algorithm (Hartley, 1997), wrapped in RANSAC.

Spec explicitly names "normalized 8 points" — implement it ourselves.
"""

import numpy as np


# 【Step 2 主函式】用 RANSAC 包裝 normalized 8-point 演算法來估計 Fundamental matrix F。
# 流程：每一輪隨機抽 8 組對應點 → 算出候選 F → 用 Sampson distance 算所有點誤差 →
# 小於 threshold 的算 inlier；跑 iters 輪後挑出 inlier 最多的那組，
# 最後再用所有 inliers 重新擬合一次 F 以提升精度。
# 作用：在含外點（錯誤匹配）的情況下穩定估計兩視角間的 epipolar 幾何。
def estimate_fundamental_ransac(pts1, pts2, threshold=1.0, iters=2000, seed=0):
    """RANSAC wrapper around the normalized 8-point F estimator.

    Args:
        pts1, pts2: (N,2) matched pixel coordinates.
        threshold:  Sampson distance (in pixels) for inlier classification.
        iters:      number of RANSAC iterations.
        seed:       RNG seed for reproducibility.

    Returns:
        F:    (3,3) fundamental matrix refit on all inliers.
        mask: (N,) bool inlier mask.
    """
    rng = np.random.default_rng(seed)
    N = pts1.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 correspondences.")

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0

    for _ in range(iters):
        idx = rng.choice(N, 8, replace=False)
        F_candidate = normalized_eight_point(pts1[idx], pts2[idx])
        if F_candidate is None:
            continue
        d = sampson_distance(F_candidate, pts1, pts2)
        inliers = d < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_count < 8:
        raise RuntimeError("RANSAC failed to find 8 consistent inliers.")

    F = normalized_eight_point(pts1[best_inliers], pts2[best_inliers])
    return F, best_inliers


# Hartley 的 normalized 8-point 演算法核心實作。
# 步驟：
#   1. 對兩組點各做相似變換（中心移到原點、平均距離縮成 sqrt(2)），得到 T1、T2。
#   2. 套用 x' = T x 把點正規化，數值上更穩定。
#   3. 由 x2'^T F' x1' = 0 建構 Nx9 的 A 矩陣。
#   4. SVD 求 A f = 0 → f 為最小奇異值對應的 V 行向量，reshape 成 3x3 F'。
#   5. 對 F' 再做一次 SVD，把最小奇異值設為 0 來強制 rank-2。
#   6. 反正規化：F = T2^T * F' * T1。
# 作用：已知 8 組以上對應點時直接解出 F，是 RANSAC 內每次抽樣的解算器。
def normalized_eight_point(pts1, pts2):
    """Normalized 8-point algorithm.

    Steps:
      1. Build similarity transforms T1, T2 that translate the centroid to 0
         and scale so the mean distance from origin is sqrt(2).
      2. Apply: x1' = T1 * x1, x2' = T2 * x2.
      3. Build A from x2'^T F' x1' = 0 (one row per correspondence).
      4. Solve A f = 0 by SVD; reshape to F'.
      5. Enforce rank-2 on F' (zero the smallest singular value).
      6. Denormalize: F = T2^T * F' * T1.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.shape[0] < 8:
        return None

    x1n, T1 = _normalize_points(pts1)
    x2n, T2 = _normalize_points(pts2)

    A = np.column_stack([
        x2n[:, 0] * x1n[:, 0], x2n[:, 0] * x1n[:, 1], x2n[:, 0],
        x2n[:, 1] * x1n[:, 0], x2n[:, 1] * x1n[:, 1], x2n[:, 1],
        x1n[:, 0],             x1n[:, 1],             np.ones(len(x1n)),
    ])
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # rank-2 constraint
    U, S, Vt2 = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm = U @ np.diag(S) @ Vt2

    F = T2.T @ F_norm @ T1
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F


# 內部 helper：對一組 2D 點做相似變換（平移 + 縮放）使其數值正規化。
# 作法：先平移把重心移到原點，再縮放使所有點到原點的平均距離為 sqrt(2)。
# 同時回傳變換矩陣 T（3x3），給後續反正規化用。
# 作用：避免座標數值差異過大導致 SVD 在解 F 時數值不穩。
def _normalize_points(pts):
    """Translate centroid to 0 and scale to mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        return pts, np.eye(3)
    s = np.sqrt(2) / mean_dist
    T = np.array([
        [s, 0, -s * centroid[0]],
        [0, s, -s * centroid[1]],
        [0, 0, 1],
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n, T


# 計算每一組對應點 (x1, x2) 對於 F 的 Sampson distance。
# Sampson distance 是真實 reprojection error 的一階近似，公式：
#     (x2^T F x1)^2 / (||(F x1)_xy||^2 + ||(F^T x2)_xy||^2)
# 作用：RANSAC 在做 inlier 判斷時用來衡量單組對應點偏離 epipolar 幾何的程度。
def sampson_distance(F, pts1, pts2):
    """First-order approximation to the geometric epipolar error."""
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    p2 = np.hstack([pts2, np.ones((len(pts2), 1))])
    Fx1 = (F @ p1.T).T              # (N,3)
    Ftx2 = (F.T @ p2.T).T           # (N,3)
    num = np.sum(p2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-12)
