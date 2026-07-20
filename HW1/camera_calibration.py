import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################

# ---------------------------------------------------------------------------
# Helper: compute Homography H for one image via DLT (Direct Linear Transform)
# ---------------------------------------------------------------------------
def compute_homography(obj_pts, img_pts):
    """
    Compute 3x3 homography H such that  λ[u,v,1]^T = H [X,Y,1]^T
    using the Direct Linear Transform (DLT) and SVD.

    obj_pts : (N,3) float  – 3-D world points (Z=0 plane)
    img_pts : (N,1,2) float – detected 2-D corner pixels
    """
    N = len(obj_pts)
    A = []
    for i in range(N):
        X, Y = float(obj_pts[i][0]), float(obj_pts[i][1])
        u, v = float(img_pts[i][0][0]), float(img_pts[i][0][1])
        # Each correspondence gives two rows in the DLT matrix
        A.append([-X, -Y, -1,  0,  0,  0, u*X, u*Y, u])
        A.append([ 0,  0,  0, -X, -Y, -1, v*X, v*Y, v])
    A = np.array(A, dtype=np.float64)

    # Solve Ah = 0  →  h = last right-singular vector (column of V)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]                       # shape (9,)
    H = h.reshape(3, 3)
    return H / H[2, 2]               # normalise so H[2,2] == 1


# ---------------------------------------------------------------------------
# Helper: v_ij vector used to build the B-matrix linear system
# ---------------------------------------------------------------------------
def vij(H, i, j):
    """
    Return the 6-D vector v_ij such that  h_i^T B h_j = v_ij^T b
    where b = [B11, B12, B22, B13, B23, B33].
    Indices i, j are 0-based column indices of H.
    """
    hi = H[:, i]   # 3-D column vector
    hj = H[:, j]
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 1+2: Compute intrinsic matrix K from a list of homographies
# ---------------------------------------------------------------------------
def compute_intrinsics(homographies):
    """
    Given N homographies, build the (2N x 6) matrix V and solve V b = 0
    for the symmetric matrix B = K^{-T} K^{-1}.
    Then recover K via Zhang's closed-form formulae (Appendix B).

    Returns K : (3,3) upper-triangular intrinsic matrix.
    """
    if len(homographies) < 3:
        raise ValueError(
            f'Need at least 3 images with detected corners, got {len(homographies)}. '
            'Check that corner_x/corner_y match your actual chessboard inner corners.'
        )
    V = []
    for H in homographies:
        # Constraint 1: h1^T B h2 = 0
        V.append(vij(H, 0, 1))
        # Constraint 2: h1^T B h1 = h2^T B h2  →  (v11 - v22)^T b = 0
        V.append(vij(H, 0, 0) - vij(H, 1, 1))
    V = np.array(V, dtype=np.float64)   # (2N, 6)

    # Solve V b = 0  →  b = last right-singular vector
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]   # [B11, B12, B22, B13, B23, B33]

    B11, B12, B22, B13, B23, B33 = b

    # Ensure B is positive-definite (flip sign if needed)
    if B11 < 0:
        B11, B12, B22, B13, B23, B33 = -b

    # Closed-form recovery of K (Zhang 2000, Appendix B)
    denom = B11 * B22 - B12 ** 2

    v0    = (B12 * B13 - B11 * B23) / denom
    lam   = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(abs(lam / B11))
    beta  = np.sqrt(abs(lam * B11 / denom))
    gamma = -B12 * alpha ** 2 * beta / lam
    u0    = gamma * v0 / beta - B13 * alpha ** 2 / lam

    K = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    return K


# ---------------------------------------------------------------------------
# Step 3: Compute extrinsic matrices [R | t] for each image
# ---------------------------------------------------------------------------
def compute_extrinsics(K, homographies):
    """
    For each homography H = K [r1, r2, t], recover:
        r1 = λ K^{-1} h1
        r2 = λ K^{-1} h2
        r3 = r1 × r2
        t  = λ K^{-1} h3
    Then project R onto SO(3) via SVD to enforce orthogonality.

    Returns extrinsics : (N, 3, 4) array
    """
    K_inv = np.linalg.inv(K)
    extrinsics = []

    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # Scale factor λ (average of two estimates for robustness)
        lam = 0.5 * (1.0 / np.linalg.norm(K_inv @ h1) +
                     1.0 / np.linalg.norm(K_inv @ h2))

        r1 =  lam * (K_inv @ h1)
        r2 =  lam * (K_inv @ h2)
        r3 =  np.cross(r1, r2)
        t  =  lam * (K_inv @ h3)

        # Project onto the nearest rotation matrix via SVD
        R_approx = np.stack([r1, r2, r3], axis=1)  # (3,3)
        U, _, Vt  = np.linalg.svd(R_approx)
        R = U @ Vt                                   # guaranteed det = +1

        ext = np.hstack([R, t.reshape(3, 1)])        # (3, 4)
        extrinsics.append(ext)

    return np.array(extrinsics)   # (N, 3, 4)


# ---------------------------------------------------------------------------
# Validation: compare our extrinsics against cv2.calibrateCamera ground truth
# ---------------------------------------------------------------------------
def validate_extrinsics(extrinsics, objpoints, imgpoints, img_size):
    """
    Compare our hand-rolled extrinsics against cv2.calibrateCamera.
    Prints per-image and average rotation (degrees) / translation (meters) errors.

    extrinsics : (N, 3, 4) array from compute_extrinsics
    objpoints  : list of (M, 3) world points per image
    imgpoints  : list of (M, 1, 2) detected corners per image
    img_size   : (width, height) tuple passed to cv2.calibrateCamera
    """
    _, _, _, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    cv2_ext = np.concatenate((np.array(rvecs), np.array(tvecs)), axis=1).reshape(-1, 6)

    errors = []
    for i in range(extrinsics.shape[0]):
        R1 = extrinsics[i, 0:3, 0:3]
        T1 = extrinsics[i, 0:3, 3]
        R2, _ = cv2.Rodrigues(cv2_ext[i, 0:3])
        T2 = cv2_ext[i, 3:6]

        rot_diff = R1.T @ R2
        angle_diff = np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1.0, 1.0))
        translation_diff = np.linalg.norm(T1 - T2)
        errors.append((angle_diff, translation_diff))

    print('Extrinsics validation vs cv2.calibrateCamera:')
    for idx, (angle_diff, translation_diff) in enumerate(errors):
        print(f'    Image {idx}: Angle Error = {np.degrees(angle_diff):.4f} deg, '
              f'Translation Error = {translation_diff:.6f} m')
    print(f'  Average: Angle = {np.degrees(np.mean([e[0] for e in errors])):.4f} deg, '
          f'Translation = {np.mean([e[1] for e in errors]):.6f} m')


# ---------------------------------------------------------------------------
# Config — change DATA_DIR to switch between datasets
# ---------------------------------------------------------------------------
DATA_DIR = 'data'          # built-in images
# DATA_DIR = 'my_data' # your own smartphone photos

# ---------------------------------------------------------------------------
# Main: data loading + calibration + visualization
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (8,6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone),
    # you need to set the corresponding numbers.
    # (7,7) for built-in data; (10,7) for my_data smartphone photos
    corner_x = 7 if DATA_DIR == 'data' else 10
    corner_y = 7
    objp = np.zeros((corner_x*corner_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(f'{DATA_DIR}/*.jpg')

    # Step through the list and search for chessboard corners
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)

        #Find the chessboard corners
        print('find the chessboard corners of',fname)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), flags)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
            plt.imshow(img)

    print('Camera calibration...')

    # Step 1 – compute one homography per image
    homographies = [
        compute_homography(obj_pts, img_pts)
        for obj_pts, img_pts in zip(objpoints, imgpoints)
    ]

    # Step 2 – recover intrinsic matrix K
    mtx = compute_intrinsics(homographies)
    print('Intrinsic matrix K:')
    print(mtx)

    # Step 3 – recover extrinsic matrices [R|t] for every image  (shape: N x 3 x 4)
    extrinsics = compute_extrinsics(mtx, homographies)
    print('Extrinsic matrices shape:', extrinsics.shape)

    # show the camera extrinsics
    print('Show the camera extrinsics')
    # plot setting
    # You can modify it for better visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # camera setting
    camera_matrix = mtx
    cam_width = 0.064/0.1
    cam_height = 0.032/0.1
    scale_focal = 1600
    # chess board setting
    board_width = 8
    board_height = 6
    square_size = 1
    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                    scale_focal, extrinsics, board_width,
                                                    board_height, square_size, True)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')
    # plt.show()

    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/result_{DATA_DIR}.jpg")

    # Validate our extrinsics against cv2.calibrateCamera
    img_size = (img.shape[1], img.shape[0])   # (width, height) from last loaded image
    validate_extrinsics(extrinsics, objpoints, imgpoints, img_size)

    #animation for rotating plot
    """
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    """
