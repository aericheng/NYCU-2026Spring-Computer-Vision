# Computer Vision — NYCU, Spring 2026

國立陽明交通大學「電腦視覺」課程作業集

## Course Info ｜ 課程資訊

- **Institution:** National Yang Ming Chiao Tung University (NYCU 國立陽明交通大學)
- **Semester:** Spring 2026（114學年度第2學期）
- **Course:** Computer Vision 電腦視覺（graduate-level course, 多媒體工程研究所）
- **Instructor:** 邱維辰 Wei-Chen Chiu
- **Field:** Computer Vision / Deep Learning

## Overview

This repository collects two homework assignments from the NYCU Computer
Vision graduate course: from-scratch camera calibration via the Direct
Linear Transform and Zhang's method (with both a static-plot and an
interactive Dash-based 3-D visualization of recovered camera poses), and a
from-scratch Structure-from-Motion pipeline (fundamental/essential matrix
estimation, RANSAC-based correspondence filtering, triangulation, and
textured 3-D mesh export). Both assignments are implemented in Python/NumPy
and OpenCV without relying on high-level calibration or SfM library
functions.
本作業集涵蓋相機校正與運動恢復結構（Structure from Motion）兩項作業,皆以
Python 從零實作核心演算法。

## Contents ｜ 內容

| Folder | Topic | Summary |
|--------|-------|---------|
| [HW1](HW1/) | Camera Calibration（相機校正） | From-scratch camera calibration using DLT + SVD (no `cv2.calibrateCamera`): recovers intrinsic matrix K and per-image extrinsics [R\|t] from chessboard images, with both a static matplotlib pipeline and an interactive Dash 3-D viewer of camera/board poses in world and camera coordinate frames. Includes a written report and results on both the provided dataset and custom smartphone-captured photos. |
| [HW2](HW2/python/) | Structure from Motion（運動恢復結構） | End-to-end SfM pipeline: correspondence matching, RANSAC fundamental matrix estimation, essential matrix decomposition and pose selection, triangulation, and textured 3-D mesh export (.obj/.mtl, viewable in Blender). Run via `python main.py mesona` / `python main.py statue` from `HW2/python/`. |

## Notes

Each `HW*/` folder keeps its own original README with setup instructions and
dependencies — please refer to the subfolder README before running any code.
These assignments were originally developed on the course's shared
repository; this repo collects only the author's own contributions (own
implementation code, report, and personally captured calibration photos),
committed here as fresh commits rather than the original shared history.
Sample/course-provided input data required to run the code (e.g. the
built-in chessboard and SfM datasets) is kept alongside the implementation
for reproducibility; the assignment spec PDFs and shared course-provided
helper scaffolding (e.g. `obj_main.py`/`CheckVisible.py`, later ported into
the author's own `HW2/python/obj_writer.py`) are not included.
