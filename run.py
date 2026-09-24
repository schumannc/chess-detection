import chess
import gradio as gr

from chess_detection import (
    CORNER_ORDER,
    annotate_hud,
    calibrate_board,
    detect_rotation,
    game_status,
    infer_move,
    move_history,
    new_stability_state,
    occupancy_from_positions,
    positions_from_occupancy,
    reinterpret,
    render_calibration,
    render_board,
    stabilize,
    track_movements,
    update_board,
)
from chess_detection.config import BOARD_ROTATION, SETTLE_FRAMES, VOTE_WINDOW
from chess_detection.logs import elide_base64_in_logs

elide_base64_in_logs()

# What the slow half of the UI is derived from. It changes only when a settled
# reading was interpreted, so its `.change` listener is what redraws the board.
BLANK_SIGNAL = (0, "", "", "", "")


def describe(squares) -> str:
    return " ".join(chess.square_name(s) for s in sorted(squares)) or "none"


def rotation_hint(board, positions, baseline, active):
    """Say whether a different grid rotation would have made sense of this change.

    A wrong rotation is the failure that looks exactly like a broken detector:
    every reading is internally consistent, it just lands on the wrong squares, so
    nothing is ever a legal move. Re-reading the same detection under the other
    three rotations settles it without touching the camera.
    """
    baseline_positions = positions_from_occupancy(baseline)
    for candidate in range(4):
        if candidate == active:
            continue
        move, _ = infer_move(
            board,
            occupancy_from_positions(reinterpret(positions, active, candidate)),
            baseline=occupancy_from_positions(
                reinterpret(baseline_positions, active, candidate)
            ),
        )
        if move is not None:
            return f"**rotation {candidate}** would read this as **{board.san(move)}**"
    return "no rotation explains it either - the detection is probably still unsettled"


def on_frame(img, points, board, stability, baseline, signal, rotation, show_squares):
    """One webcam frame: detect, wait for a settled reading, then play the move.

    Only the video and the states are outputs. The board, status, move list and
    diagnostics hang off `state_signal.change` instead, because a component listed
    as a stream output is re-rendered on every tick — `gr.skip()` does not prevent
    that, it still sends a prop update and the component still flashes.
    """
    rotation = int(rotation)
    annotated_img, positions, detections = track_movements(
        img, points, rotation=rotation, show_squares=show_squares
    )

    if not positions:
        hud = [f"detections {detections}", "not calibrated - click the 4 corners"]
        return (
            annotate_hud(annotated_img, hud),
            positions, board, stability, baseline, signal,
        )

    stability, confirmed = stabilize(stability, positions)

    if confirmed is not None:
        previous = baseline
        board, move, message, baseline = update_board(confirmed, board, previous)

        emptied = filled = hint = ""
        if previous is not None:
            detected = occupancy_from_positions(confirmed)
            emptied, filled = describe(previous - detected), describe(detected - previous)
            if move is None and message.startswith("Change does not match"):
                hint = rotation_hint(board, confirmed, previous, rotation)
        signal = (len(board.move_stack), message, emptied, filled, hint)

    hud = [
        f"detections {detections}  matched {sum(positions)}"
        f"  vote {len(stability['window'])}/{VOTE_WINDOW}"
        f"  settle {stability['count']}/{SETTLE_FRAMES}",
        f"rotation {rotation}  moves {len(board.move_stack)}",
    ]
    return (
        annotate_hud(annotated_img, hud),
        positions, board, stability, baseline, signal,
    )


def refresh_ui(board, signal, flipped):
    _, message, emptied, filled, hint = signal

    lines = []
    if message:
        lines.append(f"- last settled reading: _{message}_")
    if emptied or filled:
        lines.append(f"- emptied: `{emptied}`")
        lines.append(f"- filled: `{filled}`")
    if hint:
        lines.append(f"- {hint}")
    if not lines:
        lines.append("_Waiting for a settled reading._")
    lines.append("")
    lines.append("_Live counters are drawn on the video frame._")

    status = f"{message} - {game_status(board)}" if message else game_status(board)
    return render_board(board, flipped=flipped), status, move_history(board), "\n".join(lines)


def start_calibration(frame):
    """Freeze the current frame and drop any previous calibration.

    The clean frame is kept aside: `frozen` accumulates the corner dots and grid
    as they are clicked, and running the detector over those annotations would be
    reading the drawing rather than the board.
    """
    return (frame, frame, [], new_stability_state(), None, BLANK_SIGNAL,
            next_corner_prompt([]))


CALIBRATION_PROMPT = (
    "Click the **{corner}** corner of the playing area (corner {n} of 4). "
    "The order matters: a1 → h1 → h8 → a8, going around the board. Clicking "
    "screen corners instead mirrors the mapping, and a mirrored board reads "
    "every move as its mirror image."
)


def next_corner_prompt(points) -> str:
    if len(points) >= len(CORNER_ORDER):
        return "Calibrated."
    return CALIBRATION_PROMPT.format(corner=CORNER_ORDER[len(points)], n=len(points) + 1)


def on_corner_click(frozen_img, clean_frame, points, rotation, evt: gr.SelectData):
    """Collect a corner; on the fourth, sanity-check the result against a start position."""
    annotated, points = calibrate_board(frozen_img, points, int(rotation), evt)

    if len(points) < 4:
        return annotated, points, int(rotation), next_corner_prompt(points)
    if clean_frame is None:
        return annotated, points, int(rotation), "Calibrated."

    current = int(rotation)
    best, scores = detect_rotation(clean_frame, points)
    # Prefer to leave the rotation alone when it explains the position as well as
    # any other: clicking a1 first already determines it, and this check exists to
    # confirm the calibration, not to second-guess it.
    chosen = max(scores, key=lambda k: (scores[k], k == current, -k))

    message = (
        f"Calibrated. The pieces match a starting position on **{scores[chosen]}/64** "
        f"squares (rotation {chosen}; scores "
        + ", ".join(f"{k}:{v}" for k, v in sorted(scores.items()))
        + ")."
    )
    if scores[chosen] < 56:
        message += (
            " That is low — check the corners are on the playing area and that the "
            "board is in its starting position."
        )
    return render_calibration(clean_frame, points, chosen), points, chosen, message


def undo_move(board):
    if board.move_stack:
        board.pop()
    return board, new_stability_state(), None, (len(board.move_stack), "Move undone.", "", "", "")


def reset_game():
    board = chess.Board()
    return board, new_stability_state(), None, (0, "New game.", "", "", "")


def redraw(board, flipped):
    return render_board(board, flipped=flipped)


with gr.Blocks() as demo:
    gr.Markdown("### Chessboard calibration: freeze a frame, then click corners")

    state_points = gr.State([])                        # The 4 clicked corners
    state_positions = gr.State([])                     # The 64 occupancy bools
    state_board = gr.State(chess.Board())
    state_stability = gr.State(new_stability_state())  # Vote window + settle counter
    state_baseline = gr.State(None)                    # Last understood detection
    state_signal = gr.State(BLANK_SIGNAL)              # Drives the non-streaming UI
    state_frame = gr.State(None)                       # Clean copy of the frozen frame

    with gr.Row():
        with gr.Column():
            cam = gr.Image(label="Input", sources="webcam", streaming=True, webcam_options=gr.WebcamOptions(mirror=False))
            live = gr.Image(label="Live output (streaming)")
        with gr.Column():
            frozen = gr.Image(label="Frozen frame (click here)", interactive=False)
            calibrate_btn = gr.Button("Calibrate", variant="primary")
            board = gr.HTML(value=render_board(), label="Board")
            status = gr.Markdown("Press **Calibrate** to freeze a frame, then click the four corners.")
            history = gr.Markdown("_No moves yet._", label="Moves")

            with gr.Row():
                undo_btn = gr.Button("Undo move")
                reset_btn = gr.Button("New game")

            with gr.Accordion("Calibration options", open=False):
                rotation = gr.Radio(
                    choices=[0, 1, 2, 3],
                    value=BOARD_ROTATION,
                    label="Grid rotation (quarter-turns)",
                    info="Set automatically from the starting position when the "
                         "4th corner is clicked. If a change later matches no legal "
                         "move, the diagnostics name the rotation that would fit.",
                )
                show_squares = gr.Checkbox(
                    value=True,
                    label="Label squares on the live view",
                    info="Green = occupied. Set up a known position and read the "
                         "labels to verify the grid is aligned.",
                )
                flipped = gr.Checkbox(value=False, label="Show board from Black's side")

            with gr.Accordion("Diagnostics", open=True):
                diagnostics = gr.Markdown("_Waiting for a settled reading._")

    cam.stream(
        on_frame,
        [cam, state_points, state_board, state_stability, state_baseline, state_signal,
         rotation, show_squares],
        [live, state_positions, state_board, state_stability, state_baseline, state_signal],
        stream_every=0.5,
        time_limit=15,
        concurrency_limit=1,
    )

    # The only thing that redraws the board, and only when a reading was understood.
    state_signal.change(
        refresh_ui,
        [state_board, state_signal, flipped],
        [board, status, history, diagnostics],
    )

    calibrate_btn.click(
        start_calibration,
        cam,
        [frozen, state_frame, state_points, state_stability, state_baseline, state_signal,
         status],
    )
    frozen.select(
        on_corner_click,
        [frozen, state_frame, state_points, rotation],
        [frozen, state_points, rotation, status],
    )

    undo_btn.click(
        undo_move,
        state_board,
        [state_board, state_stability, state_baseline, state_signal],
    )
    reset_btn.click(
        reset_game,
        None,
        [state_board, state_stability, state_baseline, state_signal],
    )
    flipped.change(redraw, [state_board, flipped], board)


if __name__ == '__main__':
    demo.launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
