#import "@preview/kunskap:0.1.0": *

#set math.mat(delim: "[")
#set math.vec(delim: "[")

#show: kunskap.with(
  title: "Homework 1 — Camera Calibration",
  author: "Group 2",
  header: "NYCU - Computer Vision 2026",
  paper-size: "a4",
)
#v(-7em)

= *Introduction*

#v(-0.3em)
Camera calibration is the process of estimating the intrinsic matrix $K$ and the extrinsic matrices $[R | t]$ of a camera from images of a known pattern.
Given a set of chessboard images taken from different viewpoints, we follow *Zhang's method (Zhang et al., 2000)* to recover these parameters without expensive calibration rigs.

The core insight is that all chessboard corners lie on a *planar surface* ($Z = 0$), so the 3-D-to-2-D projection reduces to a 2-D *homography* $H$ for each image.
The calibration pipeline has three steps: (1) compute one homography $H$ per image via DLT, (2) recover the intrinsic matrix $K$ from the collection of homographies, and (3) recover the extrinsic matrices $[R | t]$ for each image.

#v(0.5em)
#figure(
  image("figures/teaser.jpg", width: 100%),
  caption: [*The concept of camera calibration.* Detected chessboard corners (left) and the recovered 3-D camera poses visualized in world coordinates (right). The board is fixed at the origin; each coloured frustum represents the camera position and orientation for one calibration image.],
)

#v(0.3em)
The assignment requires implementing all three steps from scratch, without calling `cv2.calibrateCamera`, and visualising the recovered extrinsics in 3-D.
Two datasets are used: the *provided data* folder (built-in chessboard images) and *custom photos* taken with a smartphone.

#pagebreak()

= *Implementation Procedure*

== *Step 1 — Homography via Direct Linear Transform*

#v(-0.3em)
Each image gives one $3 times 3$ homography $H$ that maps world plane coordinates $(X, Y)$ (with $Z = 0$) to pixel coordinates $(u, v)$. The projective relation is:

$
  lambda mat(u; v; 1) = H mat(X; Y; 1)
  quad => quad
  mat(lambda u; lambda v; lambda) = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) mat(X; Y; 1)
$

Expanding the matrix-vector product gives two equations:

$
  h_11 X + h_12 Y + h_13 & = underbrace(h_31 X + h_32 Y + h_33)_(lambda) u \
  h_21 X + h_22 Y + h_23 & = underbrace(h_31 X + h_32 Y + h_33)_(lambda) v
$

Rearranging so that the right-hand side is zero, and stacking both rows into a single matrix equation with $bold(h) = [h_11, h_12, h_13, h_21, h_22, h_23, h_31, h_32, h_33]^top$:

$
  mat(
    -X, -Y, -1, 0, 0, 0, u X, u Y, u;
    0, 0, 0, -X, -Y, -1, v X, v Y, v
  ) bold(h) = bold(0)
$

where $bold(h) = "vec"(H)$ is the 9-D flattened homography.
Stacking $N$ correspondences gives $A bold(h) = bold(0)$ ($A$ is $2N times 9$).
The solution is the *last right-singular vector* of $A$ (minimum singular value of SVD), which also enforces $||bold(h)|| = 1$.

```python
def compute_homography(obj_pts, img_pts):
    N = len(obj_pts)
    A = []
    for i in range(N):
        X, Y = float(obj_pts[i][0]), float(obj_pts[i][1])
        u, v = float(img_pts[i][0][0]), float(img_pts[i][0][1])
        A.append([-X, -Y, -1,  0,  0,  0, u*X, u*Y, u])
        A.append([ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v])
    A = np.array(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]           # last row of V^T = solution
    H = h.reshape(3, 3)
    return H / H[2, 2]   # normalize so H[2,2] = 1
```

#v(-0.3em)
== *Step 2 — Intrinsic Matrix $K$ via Zhang's Method*

#v(-0.3em)
The intrinsic matrix is upper-triangular with five parameters:

$
  K = mat(alpha, gamma, u_0; 0, beta, v_0; 0, 0, 1)
$

*Building the B-matrix linear system.*

Since $H = K [bold(r)_1, bold(r)_2, bold(t)]$, the columns of $H$ satisfy:

$
  mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) = K mat(r_11, r_12, t_1; r_21, r_22, t_2; r_31, r_32, t_3)
  quad => quad
  [bold(h)_1, bold(h)_2, bold(h)_3] = K [bold(r)_1, bold(r)_2, bold(t)]
  quad => quad
  K^(-1) bold(h)_1 = bold(r)_1, quad K^(-1) bold(h)_2 = bold(r)_2
$

Because $bold(r)_1$ and $bold(r)_2$ are the first two columns of a rotation matrix, they satisfy:

$
  bold(r)_1^top bold(r)_2 = 0 quad "(orthogonality)", quad
  ||bold(r)_1|| = ||bold(r)_2|| = 1 quad "(unit norm)"
$

Define the symmetric matrix $B = K^(-top) K^(-1)$, which has 6 independent entries:

$
  B = K^(-top) K^(-1)
  = mat(B_11, B_12, B_13; B_12, B_22, B_23; B_13, B_23, B_33)
  quad "(symmetric" => "6 DoF)"
$

Substituting $K^(-1) bold(h)_i = bold(r)_i$ into the two rotation constraints:

$
  bold(r)_1^top bold(r)_2 = 0 & => (K^(-1) bold(h)_1)^top (K^(-1) bold(h)_2) = 0 \
                              & => bold(h)_1^top underbrace(K^(-top) K^(-1))_(= B) bold(h)_2 = 0
                                => bold(h)_1^top B bold(h)_2 = 0
$

$
  ||bold(r)_1||^2 = ||bold(r)_2||^2
  => bold(h)_1^top B bold(h)_1 = bold(h)_2^top B bold(h)_2
$

Each homography therefore provides *two linear constraints* on the 6 entries of $B$:

$
  bold(h)_1^top B bold(h)_2 = 0 & quad => quad bold(v)_(12)^top bold(b) = 0 \
  bold(h)_1^top B bold(h)_1 = bold(h)_2^top B bold(h)_2 & quad => quad (bold(v)_(11) - bold(v)_(22))^top bold(b) = bold(0)
$

where the auxiliary 6-D vector $bold(v)_(i j)$ is:

$
  bold(v)_(i j) = mat(
    h_(i 1) h_(j 1);
    h_(i 1) h_(j 2) + h_(i 2) h_(j 1);
    h_(i 2) h_(j 2);
    h_(i 3) h_(j 1) + h_(i 1) h_(j 3);
    h_(i 3) h_(j 2) + h_(i 2) h_(j 3);
    h_(i 3) h_(j 3)
  )
$

Stacking all images gives $V bold(b) = bold(0)$ ($V$ is $2N times 6$); again solved by SVD.

```python
def vij(H, i, j):
    hi, hj = H[:, i], H[:, j]
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2],
    ])

def compute_intrinsics(homographies):
    V = []
    for H in homographies:
        V.append(vij(H, 0, 1))
        V.append(vij(H, 0, 0) - vij(H, 1, 1))
    V = np.array(V, dtype=np.float64)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]   # [B11, B12, B22, B13, B23, B33]
    if b[0] < 0: b = -b
```

*Recovering $K$ from $B$ (Zhang et al., 2000, Appendix B closed-form):*

$
     v_0 & = (B_(12) B_(13) - B_(11) B_(23)) / (B_(11) B_(22) - B_(12)^2) \
  lambda & = B_(33) - [B_(13)^2 + v_0 (B_(12) B_(13) - B_(11) B_(23))] \/ B_(11) \
   alpha & = sqrt(lambda \/ B_(11)), quad
           beta = sqrt(lambda B_(11) \/ (B_(11) B_(22) - B_(12)^2)) \
   gamma & = -B_(12) alpha^2 beta \/ lambda, quad
           u_0 = gamma v_0 \/ beta - B_(13) alpha^2 \/ lambda
$

```python
    B11, B12, B22, B13, B23, B33 = b
    denom = B11 * B22 - B12 ** 2
    v0    = (B12 * B13 - B11 * B23) / denom
    lam   = B33 - (B13**2 + v0 * (B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(abs(lam / B11))
    beta  = np.sqrt(abs(lam * B11 / denom))
    gamma = -B12 * alpha**2 * beta / lam
    u0    = gamma * v0 / beta - B13 * alpha**2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
```

#v(-0.3em)
== *Step 3 — Extrinsic Matrices $[R | t]$*

#v(-0.3em)
Given $H = K [bold(r)_1, bold(r)_2, bold(t)]$, multiplying both sides by $K^(-1)$ gives:

$
  [bold(r)_1, bold(r)_2, bold(t)] = lambda K^(-1) [bold(h)_1, bold(h)_2, bold(h)_3]
$

Since $bold(r)_1$ and $bold(r)_2$ are columns of a rotation matrix, $||bold(r)_1|| = ||bold(r)_2|| = 1$:

$
  lambda = 1 / (||K^(-1) bold(h)_1||), quad lambda = 1 / (||K^(-1) bold(h)_2||)
  quad => quad
  lambda approx 1 / 2 lr((1 / (||K^(-1) bold(h)_1||) + 1 / (||K^(-1) bold(h)_2||)))
$

$
  bold(r)_1 = lambda K^(-1) bold(h)_1, quad
  bold(r)_2 = lambda K^(-1) bold(h)_2, quad
  bold(r)_3 = bold(r)_1 times bold(r)_2, quad
  bold(t) = lambda K^(-1) bold(h)_3
$

Because corner detection introduces noise, $[bold(r)_1, bold(r)_2, bold(r)_3]$ may not be a perfect rotation matrix.
We project it onto the nearest element of $"SO"(3)$ via SVD:

$
  [U, Sigma, V^top] = "SVD"([bold(r)_1, bold(r)_2, bold(r)_3]), quad R = U V^top
$

```python
def compute_extrinsics(K, homographies):
    K_inv = np.linalg.inv(K)
    extrinsics = []
    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        lam = 0.5 * (1 / np.linalg.norm(K_inv @ h1) +
                     1 / np.linalg.norm(K_inv @ h2))
        r1 = lam * (K_inv @ h1)
        r2 = lam * (K_inv @ h2)
        r3 = np.cross(r1, r2)
        t  = lam * (K_inv @ h3)
        R_approx = np.stack([r1, r2, r3], axis=1)
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        extrinsics.append(np.hstack([R, t.reshape(3, 1)]))
    return np.array(extrinsics)   # (N, 3, 4)
```

#pagebreak()

= *Experimental Result*

== *Dataset 1 — Provided Data (`data/`)*

#v(-0.3em)
The provided dataset consists of 10 chessboard images (resolution $3916 times 2936$).
The inner corner count is *`corner_x = 7`, `corner_y = 7`* (a $7 times 7$ grid of inner corners).

#grid(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  gutter: 4pt,
  figure(image("data/0000.jpg", width: 100%)),
  figure(image("data/0001.jpg", width: 100%)),
  figure(image("data/0002.jpg", width: 100%)),
  figure(image("data/0003.jpg", width: 100%)),
  figure(image("data/0004.jpg", width: 100%)),

  figure(image("data/0005.jpg", width: 100%)),
  figure(image("data/0006.jpg", width: 100%)),
  figure(image("data/0007.jpg", width: 100%)),
  figure(image("data/0008.jpg", width: 100%)),
  figure(image("data/0009.jpg", width: 100%)),
)

#v(0.5em)
*Recovered intrinsic matrix $K$ (provided data):*

$
  K approx mat(3400.3, -35.3, 1475.7; 0, 3349.4, 1408.2; 0, 0, 1)
$

#v(0.5em)
*3-D extrinsic visualization (provided data):*

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("figures/result_world_data.png", width: 80%),
    caption: [*World Coordinate.* Board fixed, cameras moving],
  ),
  figure(
    image("figures/result_camera_data.png", width: 80%),
    caption: [*Camera Coordinate.* Camera fixed, boards moving],
  ),
)

== *Dataset 2 — Custom Smartphone Photos (`custom_data/`)*

#v(-0.3em)
Custom photos were taken with a smartphone at fixed focus (AF disabled) from 10 viewpoints.
The resolution is $3000 times 4000$ (portrait).
The inner corner count is *`corner_x = 10`, `corner_y = 7`* (a $10 times 7$ grid of inner corners on an $11 times 8$ board).

#grid(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  gutter: 4pt,
  figure(image("custom_data/0000.jpg", width: 100%)),
  figure(image("custom_data/0001.jpg", width: 100%)),
  figure(image("custom_data/0002.jpg", width: 100%)),
  figure(image("custom_data/0003.jpg", width: 100%)),
  figure(image("custom_data/0004.jpg", width: 100%)),

  figure(image("custom_data/0005.jpg", width: 100%)),
  figure(image("custom_data/0006.jpg", width: 100%)),
  figure(image("custom_data/0007.jpg", width: 100%)),
  figure(image("custom_data/0008.jpg", width: 100%)),
  figure(image("custom_data/0009.jpg", width: 100%)),
)

*Recovered intrinsic matrix $K$ (custom data):*

$
  K approx mat(3042.9, 1.4, 1556.4; 0, 3026.1, 2017.6; 0, 0, 1)
$
#v(-0.3em)

The focal length in pixels ($f_x approx f_y approx 3034$ px) is consistent with a typical smartphone lens.
The near-zero skew $gamma approx 1.4$ confirms negligible sensor shear.
The principal point $(c_x, c_y) approx (1556, 2018)$ is close to the image centre $(1500, 2000)$ as expected.

#v(-0.3em)
*3-D extrinsic visualization (custom data):*

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("figures/result_world_custom.png", width: 80%),
    caption: [*World Coordinate.* Board fixed, cameras moving],
  ),
  figure(
    image("figures/result_camera_custom.png", width: 80%),
    caption: [*Camera Coordinate.* Camera fixed, boards moving],
  ),
)

#pagebreak()

= *Discussion*

#v(-0.3em)
*Numerical stability.* The SVD-based DLT is well-conditioned as long as the corner points are spread across the image plane.
The sign ambiguity in the $B$-matrix SVD solution ($bold(b)$ vs $-bold(b)$) is resolved by checking the sign of $B_(11)$, which must be positive for a valid intrinsic matrix.

*SO(3) projection.* Direct computation of $bold(r)_1, bold(r)_2$ from the homography does not guarantee orthogonality due to noise.
The SVD-based projection $R = U V^top$ finds the nearest rotation matrix in the Frobenius norm sense.

*Custom data corner detection.* The standard `findChessboardCorners` call (without flags) failed on all 10 custom images at 1/4 resolution.
Two OpenCV flags were needed:
- `CALIB_CB_ADAPTIVE_THRESH`: applies adaptive thresholding to cope with uneven lighting.
- `CALIB_CB_NORMALIZE_IMAGE`: normalizes image intensity before detection.

*Corner count mismatch.* The provided data uses a $7 times 7$ inner-corner pattern, while the provided chessboard print (`Checkerboard-A4-25mm-10x7.png`) has a $10 times 7$ inner-corner pattern.
Mixing these configurations without changing `corner_x` / `corner_y` will cause silent detection failures.

*Comparison with `cv2.calibrateCamera` (ground-truth error analysis).*

We treat `cv2.calibrateCamera` as ground truth and compute per-image rotation and translation errors using `validate_extrinsics`.
Rotation error is the geodesic angle $theta = arccos((tr(R_1^top R_2)-1)/2)$; translation error is $||bold(t)_1 - bold(t)_2||$ in grid units.

```python
def validate_extrinsics(extrinsics, objpoints, imgpoints, img_size):
    _, _, _, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    cv2_ext = np.concatenate(
        (np.array(rvecs), np.array(tvecs)), axis=1).reshape(-1, 6)

    errors = []
    for i in range(extrinsics.shape[0]):
        R1 = extrinsics[i, 0:3, 0:3]
        T1 = extrinsics[i, 0:3, 3]
        R2, _ = cv2.Rodrigues(cv2_ext[i, 0:3])
        T2    = cv2_ext[i, 3:6]

        rot_diff  = R1.T @ R2
        angle_err = np.arccos(np.clip(
            (np.trace(rot_diff) - 1) / 2, -1.0, 1.0))
        trans_err = np.linalg.norm(T1 - T2)
        errors.append((angle_err, trans_err))
```

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header([*Dataset*], [*Avg Angle*], [*Max Angle*], [*Avg Trans*], [*Max Trans*]),
    [`data`], [5.89°], [9.50°], [3.48 units], [4.57 units],
    [`custom_data`], [2.30°], [3.63°], [0.68 units], [0.80 units],
  ),
  caption: [Extrinsic errors vs. `cv2.calibrateCamera` ground truth.],
)

Two main sources of error:

+ *No distortion model.* We assume a pure pinhole camera, ignoring radial and tangential distortion. `cv2.calibrateCamera` estimates distortion coefficients jointly with $K$ and extrinsics; without this correction, distortion is absorbed into our $R$ and $bold(t)$ estimates.

+ *No nonlinear refinement.* Our pipeline is a one-shot linear solve (DLT → B-matrix → closed-form). OpenCV uses our solution only as initialization, then refines all parameters with Levenberg-Marquardt to minimize reprojection error — a step we omit.

`custom_data` shows smaller errors (~2.3° vs. ~5.9°) because the $10 times 7$ grid provides more corners (70 vs. 49), and the smartphone lens has milder distortion.

= *Conclusion*

#v(-0.3em)
We implemented Zhang's camera calibration pipeline from scratch, including DLT-based homography estimation, B-matrix linear system construction, closed-form intrinsic recovery, and SO(3)-projected extrinsic estimation.
Both the provided dataset and custom smartphone photos were successfully calibrated.
The 3-D visualizations (world- and camera-centric) confirm physically plausible camera poses around the chessboard.

= *Submission Files*

#v(-0.3em)
#figure(
  table(
    columns: (auto, 1fr),
    align: (left + horizon, left),
    table.header([*Filename*], [*Description*]),
    [`camera_calibration.py`],
    [Implements Zhang's method: DLT homography, B-matrix intrinsics, and SO(3)-projected extrinsics.],

    [`camera_calibration_show_extrinsics.py`],
    [Plots recovered poses as 3-D frustums in world-centric and camera-centric coordinates.],

    [`camera_calibration_web.py`],
    [Gradio web interface for interactive calibration with image upload and result display.],

    [`data/` (excluded in submission)],
    [Provided dataset: 10 chessboard images ($3916 times 2936$) with $7 times 7$ inner corners.],

    [`my_data/`], [Custom smartphone photos: 10 images ($3000 times 4000$) with $10 times 7$ inner corners.],
    [`figures/`], [Figures used in this report: teaser and 3-D extrinsic visualizations for both datasets.],
    [`output/`], [Extrinsic parameters visualizations for both datasets in world coordinate.],
    [`Checkerboard-A4-25mm-10x7.png`],
    [A4 chessboard pattern ($11 times 8$ squares, 25 mm each) used for custom data collection.],
  ),
  caption: [Submitted files and their roles.],
)

= *Work Assignment Plan*

#v(-0.3em)
#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (center, center, left),
    table.header([*Member*], [*Student ID*], [*Contribution*]),
    [鄭名翔], [113950011], [Compute Homography, Report Refinement],
    [張媛婷], [113950020], [Compute Intrinsics, Report Refinement],
    [司徒立中], [111550159], [Compute Extrinsics, Report Writing],
    [何義翔], [111550106], [Validate Extrinsics, Custom Data Collection],
    [曾紹幃], [111550040], [Camera Coordinate Visualization, Report Refinement],
  ),
)
