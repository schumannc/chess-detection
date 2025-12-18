from typing import Any, Optional
import numpy as np
import supervision as sv
from PIL import Image
from .config import (
    COLOR_PALETTE,
    BBOX_MATCH_DISTANCE_THRESHOLD,
)

from .detection import detect_chess_pieces
from .calibration import grid_centers


def bbox_bottom_middle(bbox: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def bbox_distance(bbox: list[float], center: list[float]) -> float:
    c1 = bbox_bottom_middle(bbox)
    c2 = center
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def match_positions(bbox: list[list[float]], centers: list[list[float]]) -> list[list[bool]]:
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


def track_movements(img, points: list[list[float]]):
    annotated_img, detections = detect_chess_pieces(img)

    if detections is not None and len(detections) > 0:
        bbox = detections.xyxy
        bbox = bbox.tolist()
    else:
        bbox = []

    if len(points) < 4:
        # Not calibrated yet
        return annotated_img, []
    
    centers = grid_centers(points)
    positions = match_positions(bbox, centers)
    
    return annotated_img, positions