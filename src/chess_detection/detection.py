from typing import Optional, Tuple
import gradio as gr
import supervision as sv
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import torch
import numpy as np

from PIL import Image

from .config import (
    COLOR_PALETTE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DETECTION_PROMPT
)


if gr.NO_RELOAD:
    MODEL = build_sam3_image_model()
    PROCESSOR = Sam3Processor(MODEL, confidence_threshold=0.3)


def annotate(image: Image.Image, detections: sv.Detections, label: Optional[str] = None) -> Image.Image:
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)

    mask_annotator = sv.MaskAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.6
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.INDEX,
        thickness=1
    )

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(annotated_image, detections)
    annotated_image = box_annotator.annotate(annotated_image, detections)

    if label:
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.INDEX,
            text_scale=text_scale,
            text_padding=5,
            text_color=sv.Color.BLACK,
            text_thickness=1
        )
        labels = [
            f"{label} {confidence:.2f}"
            for confidence in detections.confidence
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    return annotated_image


def from_sam(sam_result: dict) -> sv.Detections:
    xyxy = sam_result["boxes"].to(torch.float32).cpu().numpy()
    confidence = sam_result["scores"].to(torch.float32).cpu().numpy()

    mask = sam_result["masks"].to(torch.bool)
    mask = mask.reshape(mask.shape[0], mask.shape[2], mask.shape[3]).cpu().numpy()

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        mask=mask
    )


def detect_chess_pieces(img) -> Tuple[np.ndarray, sv.Detections]:
    image = Image.fromarray(img).convert("RGB")
    inference_state = PROCESSOR.set_image(image)
    inference_state = PROCESSOR.set_text_prompt(state=inference_state, prompt=DETECTION_PROMPT)

    detections = from_sam(sam_result=inference_state)
    detections = detections[detections.confidence > DEFAULT_CONFIDENCE_THRESHOLD]

    annotated_image = annotate(image, detections, label=None)
    return np.array(annotated_image), detections