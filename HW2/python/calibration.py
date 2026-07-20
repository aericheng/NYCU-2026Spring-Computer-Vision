"""Chessboard calibration helper for the "your own photos" requirement.

Spec explicitly allows camera-calibration related functions (slide 1, red text),
so we use cv2.calibrateCamera here.

Usage:
    K, dist = calibrate_from_chessboard(
        glob_pattern='my_data/calib/*.jpg',
        pattern_size=(9, 6),     # inner corners per row, column
        square_size=1.0,         # arbitrary unit; only affects scale
    )
"""

from pathlib import Path
import glob
import numpy as np
import cv2


def calibrate_from_chessboard(glob_pattern, pattern_size=(9, 6), square_size=1.0):
    """Detect chessboard corners across calibration images and run cv2.calibrateCamera.

    Returns:
        K:    (3, 3) intrinsic matrix.
        dist: distortion coefficients (1, 5).
        rms:  reprojection RMS error from calibrateCamera.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points, img_points = [], []
    img_shape = None

    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No calibration images matched: {glob_pattern}")

    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            print(f"  [skip] no chessboard found in {Path(f).name}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < 3:
        raise RuntimeError("Need at least 3 valid chessboard images.")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None,
    )
    return K, dist, rms
