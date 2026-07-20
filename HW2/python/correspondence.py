"""Step 1 of the spec: find correspondences across two images using SIFT + ratio test."""

import cv2
import numpy as np


# 【Step 1】在兩張影像上各自偵測 SIFT 特徵點與描述子，再用 BFMatcher 做 KNN (k=2) 配對，
# 接著套 Lowe ratio test（最近鄰距離 < ratio * 次近鄰距離）過濾不可靠的配對。
# 最後依距離排序，回傳 (pts1, pts2) 兩個 (N,2) 的匹配像素座標陣列。
# 作用：提供後續估計 Fundamental matrix 所需要的 2D-2D 對應點。
def find_correspondences(img1, img2, ratio=0.75, max_matches=None):
    """Detect SIFT keypoints in both images and return matched (pts1, pts2).

    Args:
        img1, img2: BGR or grayscale images (np.ndarray).
        ratio:     Lowe's ratio test threshold.
        max_matches: optional cap on returned matches (sorted by distance).

    Returns:
        pts1: (N, 2) float32 in image-1 pixel coords (x, y).
        pts2: (N, 2) float32 in image-2 pixel coords (x, y).
    """
    g1 = _to_gray(img1)
    g2 = _to_gray(img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw if m.distance < ratio * n.distance]
    good.sort(key=lambda m: m.distance)
    if max_matches is not None:
        good = good[:max_matches]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


# 內部 helper：若是彩色 BGR 影像就轉成灰階，已經是灰階就直接回傳。
# 作用：SIFT 只接受單通道影像，先統一轉換。
def _to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
