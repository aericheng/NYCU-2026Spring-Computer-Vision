"""K matrix loading and small helpers."""

from pathlib import Path
import numpy as np


# 讀取助教提供的 *_calib.txt 檔案，把裡面兩台相機的內參矩陣 K1、K2 解析出來。
# 作用：跳過註解行（如 "K1:"、"Camera A:"），只擷取數字湊成兩個 3x3 矩陣，
# 並把 K[2,2] 正規化成 1（Mesona 的 K[2,2] 原本是 0.0010）。
def load_calib(path):
    """Parse the provided *_calib.txt files and return (K1, K2) as 3x3 ndarrays.

    File format example (Mesona_calib.txt):
        calibration parameters
        Camera A:
        K1:
        1.4219 0.0005 0.5092
        0 1.4219 0
        0 0 0.0010
        ...
    The Mesona K's last row is `0 0 0.0010`; we normalize so K[2,2] == 1.
    """
    nums = []
    for line in Path(path).read_text().splitlines():
        # only keep lines whose tokens are all numeric (skip "K1:", "Camera A:", etc.)
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            row = [float(t) for t in tokens]
        except ValueError:
            continue
        nums.extend(row)
    if len(nums) < 18:
        raise ValueError(f"Expected 18 numbers in {path}, got {len(nums)}")
    K1 = np.array(nums[:9], dtype=np.float64).reshape(3, 3)
    K2 = np.array(nums[9:18], dtype=np.float64).reshape(3, 3)
    return _normalize_K(K1), _normalize_K(K2)


# 內部 helper：把 K 矩陣除以 K[2,2]，讓最後一個元素變成 1（標準形式）。
def _normalize_K(K):
    if abs(K[2, 2]) < 1e-12:
        raise ValueError("K[2,2] is zero")
    return K / K[2, 2]


# 把 (N,2) 的 2D 點轉成齊次座標 (N,3)，作法是在每個點後面補一個 1。
# 用途：方便用矩陣乘法做投影、變換（x' = H @ x）。
def to_homogeneous(pts):
    """(N,2) -> (N,3) by appending 1."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.hstack([pts, np.ones((pts.shape[0], 1))])


# 把 (N,3) 齊次座標轉回 (N,2) 的歐式座標：每個點除以自己的最後一個分量。
# 用途：經過投影矩陣計算後，把齊次結果還原成像素座標。
def from_homogeneous(pts_h):
    """(N,3) -> (N,2) by dividing by last component."""
    pts_h = np.asarray(pts_h, dtype=np.float64)
    return pts_h[:, :2] / pts_h[:, 2:3]
