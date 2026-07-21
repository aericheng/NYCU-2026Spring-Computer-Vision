# HW2 - Structure from Motion (Python)

## Introduction

From-scratch, two-view Structure-from-Motion pipeline: SIFT correspondence matching, RANSAC-wrapped normalized 8-point fundamental matrix estimation, essential matrix decomposition with a cheirality test, linear (DLT) triangulation, and textured 3-D mesh export (`.obj`/`.mtl`, viewable in Blender). The core pipeline avoids high-level OpenCV helpers such as `cv2.findFundamentalMat`/`cv2.recoverPose`; `calibration.py` is a separate, spec-permitted helper that uses `cv2.calibrateCamera` for chessboard-based intrinsics on custom photos.

## Key Files

- `main.py` — pipeline entry point (Steps 1–7) and CLI argument parsing.
- `correspondence.py` — SIFT + Lowe ratio test matching (Step 1).
- `fundamental.py` — RANSAC-wrapped normalized 8-point F estimation (Step 2).
- `essential.py` — F→E conversion and cheirality-based (R, t) selection (Steps 4–5).
- `triangulation.py` — linear (DLT) triangulation (Step 6).
- `obj_writer.py` — Delaunay-based textured `.obj`/`.mtl` export (Step 7).
- `visualization.py` — epipolar line and mesh plotting.
- `utils.py` — parsing of the provided `*_calib.txt` intrinsics files.

## How to Run

```bash
pip install -r requirements.txt
python main.py mesona   # uses ../data/Mesona1.JPG, Mesona2.JPG, Mesona_calib.txt
python main.py statue   # uses ../data/Statue1.bmp, Statue2.bmp, Statue_calib.txt
```

A custom image pair can be used via `python main.py custom --img1 <path> --img2 <path> --K <path.npy>`.

## Results

Not checked into the repo — each run writes `epipolar.png`, `point_cloud.png`, `mesh.png`, and textured `model1`/`model2` `.obj`/`.mtl` files into `../output/<preset>/`.
