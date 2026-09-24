import supervision as sv
import numpy as np
import cv2
from copy import deepcopy
import gradio as gr

from .config import BOARD_GRID_SIZE, BOARD_ROTATION
from .squares import SQUARE_NAMES, reinterpret, rotation_permutation  # noqa: F401

# The corners are clicked in this order, and the order is what fixes the mapping.
# Four rotations cover only four of the eight ways a grid can sit on a board; the
# other four are mirrors, and clicking screen-corners (top-left, top-right, ...)
# lands on one whenever the camera looks from Black's side. A mirrored mapping
# reads every move as its mirror image - e2e4 becomes d2d4 - and castling stops
# resolving, because the king and queen have swapped squares. Naming each corner
# removes the choice.
CORNER_ORDER = ("a1", "h1", "h8", "a8")


def build_chessboard_edges(grid_n: int) -> list[tuple[int, int]]:
    edges = []

    def vid(r, c):
        # 1-based vertex id in row-major order
        return r * grid_n + c + 1

    # horizontal edges
    for r in range(grid_n):
        for c in range(grid_n - 1):
            edges.append((vid(r, c), vid(r, c + 1)))

    # vertical edges
    for c in range(grid_n):
        for r in range(grid_n - 1):
            edges.append((vid(r, c), vid(r + 1, c)))

    return edges


def annotate_points(img, points: list[list[float]]):
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8
    )
    annotated_img = img.copy()
    points = deepcopy(points)
    key_points = np.array(points)
    annotated_img = vertex_annotator.annotate(
        scene=annotated_img,
        key_points=sv.KeyPoints(xy=key_points[np.newaxis, ...])
    )
    # Name each corner on the frame, so a mis-ordered calibration is visible
    # rather than silently mirrored.
    for index, (x, y) in enumerate(points[:len(CORNER_ORDER)]):
        cv2.putText(annotated_img, CORNER_ORDER[index], (int(x) + 10, int(y) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_img, CORNER_ORDER[index], (int(x) + 10, int(y) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147), 1, cv2.LINE_AA)
    return annotated_img


def build_dst_grid(points: list[list[float]], rotation: int = BOARD_ROTATION) -> np.ndarray:
    src = np.array([[0,0],[BOARD_GRID_SIZE-1,0],[BOARD_GRID_SIZE-1,BOARD_GRID_SIZE-1],[0,BOARD_GRID_SIZE-1]], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    src_grid = np.array([[[x,y] for x in range(BOARD_GRID_SIZE)] for y in range(BOARD_GRID_SIZE)], dtype=np.float32)
    dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1,1,2), H).reshape(BOARD_GRID_SIZE,BOARD_GRID_SIZE,2)
    # Fix orientation: top-left → bottom-right (row-major)
    dst_grid = np.rot90(dst_grid, k=rotation)
    xy = dst_grid.reshape(-1, 2)
    return xy


def grid_centers(points: list[list[float]], rotation: int = BOARD_ROTATION) -> list[list[float]]:
    xy = build_dst_grid(points, rotation=rotation)
    centers = []
    for r in range(BOARD_GRID_SIZE - 1):
        for c in range(BOARD_GRID_SIZE - 1):
            idx1 = r * BOARD_GRID_SIZE + c
            idx2 = r * BOARD_GRID_SIZE + (c + 1)
            idx3 = (r + 1) * BOARD_GRID_SIZE + c
            idx4 = (r + 1) * BOARD_GRID_SIZE + (c + 1)
            p1 = xy[idx1]
            p2 = xy[idx2]
            p3 = xy[idx3]
            p4 = xy[idx4]
            center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
            center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
            centers.append([center_x, center_y])

    return centers


def annotate_squares(img, centers: list[list[float]], positions: list[bool] | None = None):
    """Write each square's name at its center — green when occupied, grey when empty.

    This is the only way to check the index contract without a desync going silent:
    set up a known position and read the labels off the frame.
    """
    annotated_img = img.copy()
    for idx, (cx, cy) in enumerate(centers):
        occupied = bool(positions[idx]) if positions and idx < len(positions) else False
        color = (0, 255, 0) if occupied else (160, 160, 160)
        cv2.putText(
            annotated_img,
            SQUARE_NAMES[idx],
            (int(cx) - 10, int(cy) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated_img


def annotate_hud(img, lines: list[str]):
    """Per-frame counters, drawn on the video instead of into a Markdown panel.

    The frame is replaced every tick anyway, so text on it costs no extra redraw;
    the same numbers in a Gradio component make it flash twice a second.
    """
    annotated_img = img.copy()
    for row, text in enumerate(lines):
        origin = (8, 18 + row * 16)
        cv2.putText(annotated_img, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_img, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated_img


def annotate_grid(img, points: list[list[float]], rotation: int = BOARD_ROTATION):
    xy = build_dst_grid(points, rotation=rotation)
    centers = grid_centers(points, rotation=rotation)
    centers_key_points = sv.KeyPoints(xy=np.array(centers)[np.newaxis, ...])
    edges_key_points = sv.KeyPoints(xy=xy[np.newaxis, ...])
    edges = build_chessboard_edges(BOARD_GRID_SIZE)

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2, 
        edges=edges
     )
    
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1453"),
        radius=5
    )

    annotated_img = img.copy()
    annotated_img = edge_annotator.annotate(
        scene=annotated_img,
        key_points=edges_key_points
    )
    annotated_img = vertex_annotator.annotate(
        scene=annotated_img,
        key_points=centers_key_points
    )
    return annotated_img


def render_calibration(img, points: list[list[float]], rotation: int = BOARD_ROTATION):
    """Draw whatever has been clicked so far: corners, and the grid once there are 4."""
    annotated_img = annotate_points(img, points) if points else img.copy()
    if len(points) == 4:
        annotated_img = annotate_grid(annotated_img, points, rotation=rotation)
        annotated_img = annotate_squares(
            annotated_img, grid_centers(points, rotation=rotation)
        )
    return annotated_img


def calibrate_board(img, points, rotation: int, evt: gr.SelectData):
    x, y = evt.index
    if len(points) < 4:
        points = (points or []) + [[float(x), float(y)]]
    return render_calibration(img, points, rotation), points
