from typing import Tuple, Any
import supervision as sv
import gradio as gr
import  numpy as np 
import chess
import cv2
from copy import deepcopy
from PIL import Image

from chess_detection import (
    render_board,
    calibrate_board,
    track_movements,
    update_board
)



with gr.Blocks() as demo:
    gr.Markdown("### Chessboard calibration: freeze a frame, then click corners")

    state_points = gr.State([]) # Store calibration points
    state_positions = gr.State([])  # Track piece positions
    state_board = gr.State(chess.Board())

    with gr.Row():
        with gr.Column():
            cam = gr.Image(label="Input", sources="webcam", streaming=True, webcam_options=gr.WebcamOptions(mirror=False))
            live = gr.Image(label="Live output (streaming)")
        with gr.Column():
            frozen = gr.Image(label="Frozen frame (click here)", interactive=False)
            calibrate_btn = gr.Button("Calibrate", variant="primary")
            board = gr.HTML(value=render_board(), label="Board") 

        cam.stream(
            track_movements, 
            [cam, state_points], 
            [live, state_positions], 
            stream_every=0.5, 
            time_limit=15, 
            concurrency_limit=1
        )

        # Update board
        state_positions.change(
            update_board,
            [state_positions, state_board],
            [board, state_board]
        )
        
        # Reset
        calibrate_btn.click(lambda x: (x, []), cam, [frozen, state_points])
        frozen.select(calibrate_board, [frozen, state_points], [frozen, state_points])
    

if __name__ == '__main__':
    demo.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    ) 