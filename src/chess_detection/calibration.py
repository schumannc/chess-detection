import supervision as sv
import numpy as np
import cv2
from copy import deepcopy
import gradio as gr

from .config import BOARD_GRID_SIZE


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
    return annotated_img


def build_dst_grid(points: list[list[float]]) -> np.ndarray:
    src = np.array([[0,0],[BOARD_GRID_SIZE-1,0],[BOARD_GRID_SIZE-1,BOARD_GRID_SIZE-1],[0,BOARD_GRID_SIZE-1]], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    src_grid = np.array([[[x,y] for x in range(BOARD_GRID_SIZE)] for y in range(BOARD_GRID_SIZE)], dtype=np.float32)
    dst_grid = cv2.perspectiveTransform(src_grid.reshape(-1,1,2), H).reshape(BOARD_GRID_SIZE,BOARD_GRID_SIZE,2)
    # Fix orientation: top-left → bottom-right (row-major)
    dst_grid = np.rot90(dst_grid, k=1)
    xy = dst_grid.reshape(-1, 2)
    return xy


def grid_centers(points: list[list[float]]) -> list[list[float]]:
    xy = build_dst_grid(points)
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


def annotate_grid(img, points: list[list[float]]):
    xy = build_dst_grid(points)
    centers = grid_centers(points)
    centers_key_points = sv.KeyPoints(xy=np.array(centers)[np.newaxis, ...])
    edges_key_points = sv.KeyPoints(xy=xy[np.newaxis, ...])
    edges = build_chessboard_edges(9)

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


def calibrate_board(img, points, evt: gr.SelectData):
    x, y = evt.index
    if len(points) < 4:
        points = (points or []) + [[float(x), float(y)]]
    annotated_img = annotate_points(img, points)
    if len(points) == 4:
        annotated_img = annotate_grid(annotated_img, points)
    return annotated_img, points