"""Step 6 of the spec: linear triangulation (DLT).

Given two 3x4 projection matrices P1, P2 and matched 2D points x1, x2,
solve for X (3D point) such that x1 ~ P1 X and x2 ~ P2 X.

Each correspondence yields 4 linear equations:
    [ x1 * P1[2] - P1[0] ]
    [ y1 * P1[2] - P1[1] ]   X = 0
    [ x2 * P2[2] - P2[0] ]
    [ y2 * P2[2] - P2[1] ]
Solve by SVD; X is the right-singular vector with smallest singular value.
"""

import numpy as np


# 【Step 6】線性三角化（DLT, Direct Linear Transform）。
# 給定兩台相機的投影矩陣 P1、P2 (各為 3x4) 與一組對應點 (x1, x2)，
# 求 3D 點 X 使得 x1 ~ P1 X、x2 ~ P2 X。
# 對每組對應點建構 4x4 的 A 矩陣（每張圖貢獻 2 條方程式），
# 用 SVD 解 A X = 0：X 取最小奇異值對應的右奇異向量，再除以 X[3] 還原成歐式座標。
# 作用：把 2D 對應點還原成 3D 空間中的點，產生點雲。
def triangulate(P1, P2, pts1, pts2):
    """Return (N, 3) Euclidean 3D points."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.vstack([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pts3d[i] = X[:3] / X[3]
    return pts3d


