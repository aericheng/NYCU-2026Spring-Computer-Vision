"""
Interactive web-based camera calibration visualizer.
Runs the full calibration pipeline and launches a Dash app at http://127.0.0.1:8050

Two tabs:
  - World Coordinate  : board fixed at origin, cameras move around it
  - Camera Coordinate : camera fixed at origin, boards move in front of it

Usage:
    cd HW1
    python camera_calibration_web.py
"""

import numpy as np
import cv2
import glob

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

from camera_calibration import compute_homography, compute_intrinsics, compute_extrinsics

# =============================================================================
# Config — change DATA_DIR to switch between datasets
# =============================================================================
DATA_DIR = 'data'          # built-in images
# DATA_DIR = 'my_data' # your own smartphone photos

# =============================================================================
# Calibration pipeline
# =============================================================================

corner_x = 7 if DATA_DIR == 'data' else 10
corner_y = 7
objp = np.zeros((corner_x * corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

_CB_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

objpoints, imgpoints = [], []
images = glob.glob(f'{DATA_DIR}/*.jpg')

print('Start finding chessboard corners...')
for fname in sorted(images):
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('  scanning', fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), _CB_FLAGS)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

print(f'Found corners in {len(objpoints)}/{len(images)} images.')
if len(objpoints) == 0:
    raise SystemExit(
        'No corners detected! Make sure corner_x/corner_y match your chessboard inner corners '
        f'(currently {corner_x}×{corner_y}).'
    )
print('Running camera calibration...')
homographies = [compute_homography(o, i) for o, i in zip(objpoints, imgpoints)]
mtx          = compute_intrinsics(homographies)
extrinsics   = compute_extrinsics(mtx, homographies)
print(f'K =\n{mtx}')
print(f'Extrinsics shape: {extrinsics.shape}')

# =============================================================================
# Visualization helpers
# =============================================================================

# Coordinate swap: OpenCV frame (z-forward, y-down) → plot frame (y-up, z-depth)
_SWAP = np.array([[1, 0,  0, 0],
                  [0, 0,  1, 0],
                  [0,-1,  0, 0],
                  [0, 0,  0, 1]], dtype=np.float64)

def _inv_homo(M):
    R, t = M[:3, :3], M[:3, 3]
    M_inv = np.eye(4)
    M_inv[:3, :3] = R.T
    M_inv[:3, 3]  = -R.T @ t
    return M_inv

def _apply(M, pts_4xN, inverse=False):
    T = _inv_homo(M) if inverse else M
    return _SWAP @ T @ pts_4xN


def _camera_model(K, cam_w, cam_h, scale_focal):
    """Return list of (4×N) homogeneous point arrays for a camera frustum."""
    fx, fy = K[0, 0], K[1, 1]
    f = scale_focal * 2.0 / (fx + fy)
    w, h = cam_w, cam_h

    img_plane = np.array([[-w, h, f, 1], [ w, h, f, 1],
                           [ w,-h, f, 1], [-w,-h, f, 1],
                           [-w, h, f, 1]], dtype=np.float64).T
    triangle  = np.array([[-w,-h, f, 1], [ 0,-2*h, f, 1],
                           [ w,-h, f, 1]], dtype=np.float64).T

    def ray(px, py):
        return np.array([[0, 0, 0, 1], [px, py, f, 1]], dtype=np.float64).T

    return [img_plane, triangle, ray(-w, h), ray(w, h), ray(w,-h), ray(-w,-h)]


def _board_model(board_w, board_h, sq):
    """Return (4×5) homogeneous points for the chessboard outline."""
    w, h = board_w * sq, board_h * sq
    pts = np.array([[0,0,0,1],[w,0,0,1],[w,h,0,1],[0,h,0,1],[0,0,0,1]],
                   dtype=np.float64).T
    return [pts]


def build_traces(K, extrinsics, cam_w, cam_h, scale_focal,
                 board_w, board_h, sq, pattern_centric):
    """
    Build Plotly Scatter3d traces.
    pattern_centric=True  → World coord  (board fixed, cameras move)
    pattern_centric=False → Camera coord (camera fixed, boards move)
    """
    n = len(extrinsics)
    colors = px.colors.sample_colorscale('Jet', [i / max(n - 1, 1) for i in range(n)])

    static_geoms = (_board_model(board_w, board_h, sq)   if pattern_centric
                    else _camera_model(K, cam_w, cam_h, scale_focal))
    moving_geoms = (_camera_model(K, cam_w, cam_h, scale_focal) if pattern_centric
                    else _board_model(board_w, board_h, sq))

    traces = []

    for pts in static_geoms:
        p = _apply(np.eye(4), pts)
        traces.append(go.Scatter3d(
            x=p[0], y=p[1], z=p[2], mode='lines',
            line=dict(color='red', width=3),
            showlegend=False, hoverinfo='skip',
        ))

    for i, ext in enumerate(extrinsics):
        cMo = np.eye(4)
        cMo[:3, :3] = ext[:3, :3]
        cMo[:3, 3]  = ext[:3, 3]
        first = True
        for pts in moving_geoms:
            p = _apply(cMo, pts, inverse=pattern_centric)
            traces.append(go.Scatter3d(
                x=p[0], y=p[1], z=p[2], mode='lines',
                line=dict(color=colors[i], width=2),
                name=f'Image {i}', showlegend=first,
                legendgroup=f'img{i}',
                hovertemplate=f'Image {i}<extra></extra>',
            ))
            first = False

    return traces

# =============================================================================
# Dash application
# =============================================================================

CAM_W, CAM_H = 0.064 / 0.1, 0.032 / 0.1
SCALE_FOCAL  = 1600
BOARD_W, BOARD_H, SQ_SIZE = 8, 6, 1

_TAB_STYLE          = {'fontFamily': 'Arial', 'fontSize': '15px', 'padding': '10px 20px'}
_TAB_SELECTED_STYLE = {**_TAB_STYLE, 'fontWeight': 'bold', 'borderTop': '3px solid #2196F3'}

app = dash.Dash(__name__, title='Camera Calibration Viewer')

app.layout = html.Div(style={'fontFamily': 'Arial'}, children=[
    html.H2('Camera Calibration — Extrinsic Parameters Visualization',
            style={'textAlign': 'center', 'marginBottom': '4px'}),
    html.P(f'Calibrated from {len(extrinsics)} images  |  '
           f'fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f}  '
           f'cx={mtx[0,2]:.1f}  cy={mtx[1,2]:.1f}',
           style={'textAlign': 'center', 'color': '#555', 'marginTop': '0'}),

    dcc.Tabs(id='coord-tabs', value='world',
             style={'marginBottom': '4px'}, children=[
        dcc.Tab(label='🌍  World Coordinate  (Fix Board, Move Cameras)',
                value='world', style=_TAB_STYLE, selected_style=_TAB_SELECTED_STYLE),
        dcc.Tab(label='📷  Camera Coordinate  (Fix Camera, Move Boards)',
                value='camera', style=_TAB_STYLE, selected_style=_TAB_SELECTED_STYLE),
    ]),

    dcc.Graph(id='extrinsic-plot', style={'height': '82vh'},
              config={'displayModeBar': True, 'scrollZoom': True}),
])


@app.callback(Output('extrinsic-plot', 'figure'), Input('coord-tabs', 'value'))
def update_figure(tab_value):
    pattern_centric = (tab_value == 'world')
    traces = build_traces(mtx, extrinsics, CAM_W, CAM_H, SCALE_FOCAL,
                          BOARD_W, BOARD_H, SQ_SIZE, pattern_centric)

    title = ('World Coordinate — Board Fixed, Cameras Moving' if pattern_centric
             else 'Camera Coordinate — Camera Fixed, Boards Moving')

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(xaxis_title='X', yaxis_title='Z (depth)', zaxis_title='−Y',
                   aspectmode='data'),
        legend=dict(title='Images', x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


if __name__ == '__main__':
    print('\nStarting web server — open http://127.0.0.1:8050\n')
    app.run(debug=False)
