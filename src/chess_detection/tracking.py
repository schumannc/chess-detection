from typing import Any, Optional
import numpy as np
import supervision as sv
from PIL import Image
from .config import (
    COLOR_PALETTE,
    BBOX_MATCH_DISTANCE_THRESHOLD
)

from .detection import detect_chess_pieces


def bbox_bottom_middle(bbox: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def bbox_distance(bbox1: list[float], bbox2: list[float]) -> float:
    c1 = bbox_bottom_middle(bbox1)
    c2 = bbox_bottom_middle(bbox2)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def match_bboxes(prev_bboxes: list[list[float]], curr_bboxes: list[list[float]]) -> dict[int, int]:
    matches = {}
    used_prev = set()

    if not prev_bboxes:
        return {i: -1 for i in range(len(curr_bboxes))}

    for curr_idx, curr_bboxes in enumerate(curr_bboxes):
        best_match = -1
        min_distance = BBOX_MATCH_DISTANCE_THRESHOLD
        for prev_idx, prev_bbox in enumerate(prev_bboxes):
            if prev_idx in used_prev:
                continue
            
            dist = bbox_distance(curr_bboxes, prev_bbox)
            if dist < min_distance:
                best_match = prev_idx
                min_distance = dist

        if best_match != -1:
            used_prev.add(best_match)

        matches[curr_idx] = best_match
        
    return matches    


def track_movements(img, prev_bboxes: Any):
    annotated_img, detections = detect_chess_pieces(img)

    if detections is not None and len(detections) > 0:
        bbox = detections.xyxy
        curr_bbox = bbox.tolist()
    else:
        curr_bbox = []

    matches = match_bboxes(prev_bboxes, curr_bbox)

    # Annotad moved pieces
    for curr_idx, prev_idx in matches.items():
        if prev_idx != -1:
            continue
        
        image = Image.fromarray(annotated_img).convert("RGB")
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.INDEX,
            text_scale=text_scale,
            text_padding=5,
            text_color=sv.Color.BLACK,
            text_thickness=1
        )

        annotated_img = label_annotator.annotate(
            scene=annotated_img,
            detections=detections[[curr_idx]],
            labels=["moved"]
        )

    return annotated_img, curr_bbox, matches