from typing import Any, Optional
import numpy as np
import supervision as sv
from PIL import Image
from .config import (
    BBOX_MATCH_DISTANCE_THRESHOLD,
    BOARD_ROTATION,
)

from .board import START_OCCUPANCY
from .detection import detect_chess_pieces
from .stability import new_stability_state, stabilize  # noqa: F401  (re-export)
from .calibration import grid_centers, annotate_grid, annotate_squares


def bbox_bottom_middle(bbox: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def bbox_distance(bbox: list[float], center: list[float]) -> float:
    c1 = bbox_bottom_middle(bbox)
    c2 = center
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def match_positions(bbox: list[list[float]], centers: list[list[float]]) -> list[bool]:
    matches = [False for _ in range(len(centers))]
    used_centers = set()

    for b in bbox:
        best_center_idx: Optional[int] = None
        best_distance = float('inf')
        for idx, c in enumerate(centers):
            if idx in used_centers:
                continue
            dist = bbox_distance(b, c)
            if dist < best_distance:
                best_distance = dist
                best_center_idx = idx
        if best_center_idx is not None and best_distance < BBOX_MATCH_DISTANCE_THRESHOLD:
            matches[best_center_idx] = True
            used_centers.add(best_center_idx)
        
    return matches    


def track_movements(
    img,
    points: list[list[float]],
    rotation: int = BOARD_ROTATION,
    show_squares: bool = True,
) -> tuple[np.ndarray, list[bool], int]:
    annotated_img, detections = detect_chess_pieces(img)

    if detections is not None and len(detections) > 0:
        bbox = detections.xyxy.tolist()
    else:
        bbox = []

    if len(points) < 4:
        # Not calibrated yet
        return annotated_img, [], len(bbox)

    centers = grid_centers(points, rotation=rotation)
    positions = match_positions(bbox, centers)

    annotated_img = annotate_grid(annotated_img, points, rotation=rotation)
    if show_squares:
        annotated_img = annotate_squares(annotated_img, centers, positions)

    return annotated_img, positions, len(bbox)


def detect_rotation(img, points: list[list[float]]) -> tuple[int, dict[int, int]]:
    """Pick the grid rotation that makes the pieces look like a starting position.

    The rotation is the one setting nothing else can verify — a wrong value is
    self-consistent, it just lands every piece on the wrong square, so the symptom
    is "no change is ever a legal move" rather than an error. Scoring all four
    against a known position removes the guesswork.

    Rotations 0 and 2 always tie, because a start position is unchanged by a half
    turn; the tie is broken toward 0, which puts rank 1 on the edge away from the
    camera (White sitting opposite). Both are returned in `scores` so the caller
    can say so.
    """
    _, detections = detect_chess_pieces(img)
    bbox = detections.xyxy.tolist() if detections is not None and len(detections) else []

    scores = {}
    for candidate in range(4):
        positions = match_positions(bbox, grid_centers(points, rotation=candidate))
        occupied = {i for i, occupied_here in enumerate(positions) if occupied_here}
        scores[candidate] = 64 - len(occupied ^ START_OCCUPANCY)

    best = max(scores, key=lambda candidate: (scores[candidate], -candidate))
    return best, scores
