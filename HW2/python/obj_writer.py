"""Step 7 of the spec: write a textured 3D mesh as .obj + .mtl.

Python port of the provided MATLAB obj_main.m / CheckVisible.m, with:
  - 2D Delaunay triangulation on image-2 points
  - Back-face winding-order flip via the camera-direction dot product test
  - Texture coords normalized to [0,1] with v flipped (1 - y/H)
"""

from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import Delaunay


# 【Step 7】把 3D 點雲輸出成有貼圖的 .obj / .mtl，方便後續用 Blender 開啟（Step 8）。
# 流程：
#   1. 在 2D 影像上的特徵點做 Delaunay 三角化，得到三角面片 (M,3) 索引。
#   2. 寫 .obj：先寫所有頂點 (v x y z)，再寫紋理座標 (vt u v)
#      （u = x/W、v = 1 - y/H，因為 OBJ 的 v 軸向上、影像 y 軸向下要翻轉）。
#   3. 寫面片 (f i/i j/j k/k)：用相機光軸方向 cam_dir = M[2,:3] 與三角形法向量
#      做點積，<0 代表面朝鏡頭（visible），保留原順序；>=0 代表背面，把前兩個頂點交換
#      讓它「翻過來」朝向鏡頭（back-face winding flip）。
#   4. 寫對應的 .mtl，把材質 map_Kd 指向紋理影像，並把該影像複製一份到 output_dir，
#      讓 Blender 可以直接找到貼圖。
# 作用：把純 3D 點雲變成「有顏色 / 紋理」的 3D 模型，供視覺化與報告展示。
def write_textured_obj(pts3d, pts2d_img2, M, tex_path, output_dir, im_index):
    """Write model{im_index}.obj and model{im_index}.mtl into output_dir.

    Args:
        pts3d:      (N, 3) 3D vertices.
        pts2d_img2: (N, 2) 2D pixel coords on the texture image (image 2).
        M:          (3, 4) projection matrix of the texture camera (= K2 [R|t]).
                    Used for back-face check.
        tex_path:   path to texture image file (used in .mtl as map_Kd).
        output_dir: where to write .obj / .mtl / mesh plot.
        im_index:   integer suffix for filenames.

    Returns:
        triangles:  (M, 3) array of vertex indices used (after Delaunay).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(tex_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read texture image: {tex_path}")
    H, W = img.shape[:2]

    tri = Delaunay(pts2d_img2).simplices  # (M, 3) vertex indices

    obj_path = output_dir / f"model{im_index}.obj"
    mtl_path = output_dir / f"model{im_index}.mtl"

    with open(obj_path, "w") as fid:
        fid.write("# obj file\n")
        fid.write(f"mtllib {mtl_path.name}\n\n")
        fid.write("usemtl Texture\n")

        # vertices
        for v in pts3d:
            fid.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        fid.write("\n\n\n")

        # texture coords (v flipped, OBJ convention)
        for p in pts2d_img2:
            u = p[0] / W
            v = 1.0 - p[1] / H
            fid.write(f"vt {u:.6f} {v:.6f}\n")
        fid.write("\n\n\n")

        # faces (1-indexed, with back-face flip)
        cam_dir = M[2, :3]
        for t in tri:
            a, b, c = pts3d[t[0]], pts3d[t[1]], pts3d[t[2]]
            normal = np.cross(b - a, c - b)
            visible = np.dot(cam_dir, normal) < 0
            if visible:
                i, j, k = t[0] + 1, t[1] + 1, t[2] + 1
            else:
                i, j, k = t[1] + 1, t[0] + 1, t[2] + 1
            fid.write(f"f {i}/{i} {j}/{j} {k}/{k}\n")

    with open(mtl_path, "w") as fmtl:
        fmtl.write("# MTL file\n")
        fmtl.write("newmtl Texture\n")
        fmtl.write("Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n")
        # texture path: copy image into output_dir for portability
        tex_basename = Path(tex_path).name
        fmtl.write(f"map_Kd {tex_basename}\n")

    # copy texture next to the .obj/.mtl so Blender finds it
    dst = output_dir / Path(tex_path).name
    if not dst.exists():
        cv2.imwrite(str(dst), img)

    return tri
