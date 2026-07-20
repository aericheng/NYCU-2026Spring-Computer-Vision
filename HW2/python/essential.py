"""Steps 4 & 5 of the spec:
  - Compute essential matrix E from F (E = K2^T F K1) and enforce
    the two equal singular values constraint exactly as the spec hint shows.
  - Decompose E into the 4 candidate (R, t) pairs.
  - Pick the unique physically-correct pair via the cheirality test
    (positive depth in BOTH cameras for the most points).
"""

import numpy as np
from triangulation import triangulate


# 【Step 4-1】把 Fundamental matrix F 轉成 Essential matrix E。
# 公式：E = K2^T F K1（用兩台相機的內參把像素座標下的 F 轉到歸一化座標下的 E）。
# 接著做 SVD 並強制兩個非零奇異值相等（取平均 m），第三個奇異值設為 0：
#     E = U * diag([m, m, 0]) * V^T
# 作用：讓 E 嚴格符合 essential matrix 應有的代數性質，後續分解 (R, t) 才會穩定。
def fundamental_to_essential(F, K1, K2):
    """E = K2^T F K1, then enforce equal-singular-value constraint
    (spec hint, slide 3):

        [U,S,V] = svd(E);
        m = (S(1,1)+S(2,2))/2;
        E = U * diag([m, m, 0]) * V'
    """
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2
    E = U @ np.diag([m, m, 0.0]) @ Vt
    return E


# 【Step 4-2】把 Essential matrix E 分解成 4 組 (R, t) 候選解。
# 作法：對 E 做 SVD 得到 U、V^T；用旋轉輔助矩陣 W 構造兩種旋轉：R1 = U W V^T、R2 = U W^T V^T，
# 平移方向 t = ±U[:,2]（U 第三行），共 2x2 = 4 組 (R, ±t)。
# 中間若 U 或 V^T 的 det 為負會翻號，避免得到反射（improper rotation）。
# 作用：因為 E 只能解到 sign 與旋轉組合的歧義，必須把 4 個候選都列出，下一步再挑唯一正解。
def decompose_essential(E):
    """Return the 4 candidate (R, t) decompositions of E.

    Spec hint (slide 3):
        W = [[0,-1,0],[1,0,0],[0,0,1]]
        R = U W V^T  or  U W^T V^T
        t = ±U[:,2]
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=np.float64)
    R1 = U @ W   @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)
    return [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]


# 【Step 5】從 4 組 (R, t) 候選中挑出物理上正確的那一組（cheirality test）。
# 對每個候選都做一次 triangulation 得到 3D 點，然後檢查每個點：
#   z1 > 0：在相機 1 的前方（相機 1 設在世界原點，z 軸即光軸方向）
#   z2 > 0：經過 [R|t] 變到相機 2 座標系後仍在相機前方
# 兩條件都滿足的點數最多者就是正解。
# 回傳：選中的 R, t、兩台相機的投影矩陣 P1、P2、以及通過測試的 3D 點。
# 作用：解決 E 分解的 4-fold 歧義，得到唯一的兩視角相對位姿。
def select_pose(candidates, K1, K2, pts1, pts2):
    """Pick the (R, t) under which the most triangulated points have
    positive depth in BOTH cameras (cheirality test).

    Returns:
        R, t (with t shape (3,1))
        P1   (3x4) projection matrix of camera 1 in world frame: K1 [I | 0]
        P2   (3x4) projection matrix of camera 2 in world frame: K2 [R | t]
        pts3d_inliers (M, 3) the triangulated points that passed cheirality
                              for the chosen pose.
    """
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    best_count = -1
    best_pts3d = None
    for R, t in candidates:
        P2 = K2 @ np.hstack([R, t])
        pts3d = triangulate(P1, P2, pts1, pts2)        # (N,3) in world (= cam-1) frame
        z1 = pts3d[:, 2]
        # depth in camera 2:
        pts3d_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        z2 = (np.hstack([R, t]) @ pts3d_h.T)[2]
        good = (z1 > 0) & (z2 > 0)
        count = int(good.sum())
        if count > best_count:
            best_count = count
            best = (R, t, P1, P2)
            best_pts3d = pts3d[good]

    R, t, P1, P2 = best
    return R, t, P1, P2, best_pts3d
