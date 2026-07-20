"""Step 3 (epipolar lines) and Step 7 (mesh plot) visualizations."""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


# 【Step 3】畫 epipolar lines 用來視覺化驗證 F 的正確性。
# 流程：在所有 inlier 中隨機選 n_show 個點，左邊圖 img1 上畫圓點標記特徵點，
# 右邊圖 img2 上畫對應的 epipolar line：l2 = F @ x1（齊次直線 ax+by+c=0 形式），
# 並在 img2 上同色畫出該特徵點實際匹配位置。
# 作用：若 F 估得正確，特徵點應該幾乎落在對應的 epipolar line 上（肉眼即可檢查）。
def draw_epipolar(img1, img2, pts1, pts2, F, out_path, n_show=30, seed=0):
    """Draw `n_show` interest points on img1 and their epipolar lines on img2.

    Epipolar line in image 2 corresponding to x1 is l2 = F @ x1.
    Epipolar line in image 1 corresponding to x2 is l1 = F.T @ x2.
    Saves a side-by-side composite to out_path.
    """
    rng = np.random.default_rng(seed)
    n = min(n_show, len(pts1))
    idx = rng.choice(len(pts1), n, replace=False)

    vis1 = _ensure_color(img1).copy()
    vis2 = _ensure_color(img2).copy()
    h2, w2 = vis2.shape[:2]
    h1, w1 = vis1.shape[:2]

    for i in idx:
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
        x2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])

        # interest point on img1
        cv2.circle(vis1, (int(pts1[i, 0]), int(pts1[i, 1])), 4, color, -1)

        # epipolar line on img2: l = F @ x1, l = (a,b,c) so ax+by+c=0
        a, b, c = F @ x1
        if abs(b) > 1e-8:
            x0_, y0_ = 0, int(-c / b)
            x1_, y1_ = w2 - 1, int(-(c + a * (w2 - 1)) / b)
            cv2.line(vis2, (x0_, y0_), (x1_, y1_), color, 1)
        cv2.circle(vis2, (int(pts2[i, 0]), int(pts2[i, 1])), 4, color, -1)

    composite = _hstack_pad(vis1, vis2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)


# 【Step 7 視覺化】用 matplotlib 的 plot_trisurf 畫出 3D 三角網格（mesh）。
# 輸入：pts3d 是 (N,3) 頂點座標、triangles 是 (M,3) 三角形面片的頂點索引。
# 作用：作業規格要求要附上 mesh 圖（不只是點雲），這個函式存出 mesh.png。
def plot_mesh(pts3d, triangles, out_path, title="3D mesh"):
    """Save a trisurf-style mesh plot (matplotlib)."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
                    triangles=triangles, cmap='viridis',
                    edgecolor='gray', linewidth=0.2, alpha=0.8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# 【Step 6 視覺化】把三角化後的 3D 點以散布圖形式畫成點雲。
# 用途：在跑出 mesh 之前先確認三角化結果合理（無離群、形狀大致正確）。
def plot_point_cloud(pts3d, out_path, title="3D points"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=4)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# 內部 helper：確保影像是 3 通道彩色（給 cv2.circle/line 畫彩色標記用），
# 灰階圖會被轉成 BGR。
def _ensure_color(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# 內部 helper：把兩張高度可能不同的影像水平拼成一張（短邊在底部補 0）。
# 作用：把 img1（特徵點）與 img2（epipolar lines）並排輸出成單一視覺化結果。
def _hstack_pad(a, b):
    h = max(a.shape[0], b.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        out = np.zeros((h, im.shape[1], 3), dtype=im.dtype)
        out[:im.shape[0]] = im
        return out
    return np.hstack([pad(a), pad(b)])
