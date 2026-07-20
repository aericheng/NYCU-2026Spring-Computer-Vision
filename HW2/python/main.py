"""End-to-end SfM pipeline (HW2).

Run from the python/ directory:
    python main.py mesona     # uses ../data/Mesona1.JPG, Mesona2.JPG, Mesona_calib.txt
    python main.py statue
    python main.py custom --img1 ../my_data/a.jpg --img2 ../my_data/b.jpg \
                          --K ../my_data/K.npy

Outputs (in ../output/<name>/):
    epipolar.png            Step 3 visualization
    point_cloud.png         Step 6 visualization
    mesh.png                Step 7 mesh plot (mandatory per spec)
    model{1,2}.obj/.mtl     Step 7 textured 3D model (load in Blender, Step 8)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from utils import load_calib
from correspondence import find_correspondences
from fundamental import estimate_fundamental_ransac
from essential import fundamental_to_essential, decompose_essential, select_pose
from triangulation import triangulate
from visualization import draw_epipolar, plot_mesh, plot_point_cloud
from obj_writer import write_textured_obj


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
MY_DATA = ROOT / "my_data"
OUTPUT = ROOT / "output"


PRESETS = {
    "mesona": {
        "img1": DATA / "Mesona1.JPG",
        "img2": DATA / "Mesona2.JPG",
        "calib": DATA / "Mesona_calib.txt",
    },
    "statue": {
        "img1": DATA / "Statue1.bmp",
        "img2": DATA / "Statue2.bmp",
        "calib": DATA / "Statue_calib.txt",
    },
}


# 解析命令列參數。支援三種 preset：mesona / statue / custom。
# custom 模式需要使用者自行提供 --img1、--img2、--K (.npy 內參矩陣)。
# 也提供 RANSAC 與 SIFT 比對的可調參數（threshold、iterations、ratio、max_matches）。
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=list(PRESETS.keys()) + ["custom"])
    p.add_argument("--img1", type=Path)
    p.add_argument("--img2", type=Path)
    p.add_argument("--K", type=Path, help="path to .npy with 3x3 K (custom only)")
    p.add_argument("--ransac-thresh", type=float, default=1.0)
    p.add_argument("--ransac-iters", type=int, default=2000)
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    p.add_argument("--max-matches", type=int, default=2000)
    return p.parse_args()


# 根據 args 解析出實際要使用的兩張影像路徑與兩台相機的內參 K1、K2。
# preset = "custom" 時直接從 .npy 載入（兩台相機共用同一個 K）；
# 否則從 PRESETS dict 找預設路徑，並用 utils.load_calib 解析 *_calib.txt。
def resolve_inputs(args):
    if args.preset == "custom":
        if not (args.img1 and args.img2 and args.K):
            sys.exit("custom preset requires --img1 --img2 --K")
        K = np.load(args.K)
        return args.img1, args.img2, K, K, args.preset
    cfg = PRESETS[args.preset]
    K1, K2 = load_calib(cfg["calib"])
    return cfg["img1"], cfg["img2"], K1, K2, args.preset


# 整個 SfM pipeline 的進入點，依序串起作業規格的 Step 1~7：
#   Step 1：SIFT + Lowe ratio test 找對應點。
#   Step 2：normalized 8-point + RANSAC 估計 Fundamental matrix F。
#   Step 3：畫 epipolar lines 視覺化驗證 F。
#   Step 4：F → E（用 K1, K2 轉換），SVD 分解出 4 組 (R, t) 候選。
#   Step 5：cheirality test 從 4 組中挑出物理上正確的 (R, t)。
#   Step 6：用挑到的位姿做三角化，再過濾一次保留兩台相機都正深度的點，輸出點雲圖。
#   Step 7：對 image1、image2 各做一次 Delaunay + 紋理貼圖，輸出 .obj/.mtl 與 mesh 圖。
# 所有輸出存到 ../output/<preset>/。
def main():
    args = parse_args()
    img1_path, img2_path, K1, K2, name = resolve_inputs(args)
    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {img1_path.name}, {img2_path.name}")
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        sys.exit(f"Failed to read {img1_path} or {img2_path}")

    print("[step 1] correspondences (SIFT + Lowe ratio)")
    pts1, pts2 = find_correspondences(
        img1, img2, ratio=args.ratio, max_matches=args.max_matches,
    )
    print(f"         got {len(pts1)} matches")

    print("[step 2] normalized 8-point + RANSAC -> F")
    F, inliers = estimate_fundamental_ransac(
        pts1, pts2,
        threshold=args.ransac_thresh,
        iters=args.ransac_iters,
    )
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    print(f"         {inliers.sum()}/{len(pts1)} inliers")

    print("[step 3] draw epipolar lines")
    draw_epipolar(img1, img2, pts1_in, pts2_in, F, out_dir / "epipolar.png")

    print("[step 4] F -> E and 4 candidate (R, t)")
    E = fundamental_to_essential(F, K1, K2)
    candidates = decompose_essential(E)

    print("[step 5] cheirality test -> pick (R, t)")
    R, t, P1, P2, pts3d = select_pose(candidates, K1, K2, pts1_in, pts2_in)
    print(f"         R = \n{R}\n         t = {t.ravel()}")
    print(f"         {len(pts3d)} points passed cheirality")

    # keep only the inlier 2D points whose 3D had positive depth in both cams.
    # select_pose currently returns the cheirality-filtered 3D points but not
    # which inlier indices they came from; re-filter here using the same test.
    z1 = (np.linalg.inv(K1) @ np.hstack([pts1_in, np.ones((len(pts1_in), 1))]).T)
    P1_world = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    pts3d_all = triangulate(P1_world, K2 @ np.hstack([R, t]), pts1_in, pts2_in)
    pts3d_h = np.hstack([pts3d_all, np.ones((len(pts3d_all), 1))])
    z1 = pts3d_all[:, 2]
    z2 = (np.hstack([R, t]) @ pts3d_h.T)[2]
    good = (z1 > 0) & (z2 > 0)
    pts3d = pts3d_all[good]
    pts1_kept = pts1_in[good]
    pts2_kept = pts2_in[good]

    print("[step 6] triangulation done — visualizing point cloud")
    plot_point_cloud(pts3d, out_dir / "point_cloud.png",
                     title=f"{name}: 3D points (n={len(pts3d)})")

    print("[step 7] texture mapping -> .obj/.mtl + mesh plot")
    # write two models: textured by image 1 and image 2 (mirrors obj_main calls).
    M1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M2 = K2 @ np.hstack([R, t])
    tri1 = write_textured_obj(pts3d, pts1_kept, M1, img1_path, out_dir, im_index=1)
    tri2 = write_textured_obj(pts3d, pts2_kept, M2, img2_path, out_dir, im_index=2)
    plot_mesh(pts3d, tri2, out_dir / "mesh.png", title=f"{name}: mesh")

    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
