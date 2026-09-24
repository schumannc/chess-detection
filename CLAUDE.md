# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build the image and run the Gradio app (http://localhost:7860)
docker compose up --build

# Shell inside the running container
docker compose exec app bash

# One-off command (container does not need to be up)
docker compose run --rm app python -c "import torch; print(torch.cuda.is_available())"
```

Outside Docker, the package needs an editable install first — the src layout means
run.py's `from chess_detection import ...` fails without it:

```bash
pip install -e . && python run.py
```

The occupancy-to-move logic has a test script that needs neither a camera nor a GPU —
`python-chess` alone is enough, so it runs on the host:

```bash
pip install chess && python tests/test_inference.py
```

It exits non-zero on failure. `tests/test_mapping.py` covers the calibration geometry and needs
opencv, so it runs in the container:

```bash
docker compose run --rm app python tests/test_mapping.py
```

There are no linters or formatters configured in this repo.

Note: the README says `python -m app.app`, but there is no `app/` package — `run.py` is the real entrypoint.

## Dependencies

`requirements.txt` is deliberately partial. The heavy dependencies — `sam3` (cloned from
github.com/facebookresearch/sam3 and pip-installed editable), plus the `torch` / `opencv` /
`numpy` stack it pulls in — are installed by the root `Dockerfile`, not by pip from this repo.
Outside Docker, SAM3 and CUDA torch must be installed by hand.

The image builds on `nvidia/cuda:12.1.1` and uses micromamba to get Python 3.11 (Ubuntu 22.04
ships 3.10). `HF_TOKEN` (see `.env.example`) is needed for SAM3 weight downloads; compose reads
it from `.env`.

Inside the container the project is *not* pip-installed. `PYTHONPATH=/app/src` (set in the
Dockerfile) is what makes `import chess_detection` work, so nothing writes `*.egg-info` back
into the bind-mounted working tree. Downloaded weights live in the `hf-cache` named volume
(`HF_HOME=/cache/huggingface`), which is why a rebuild does not re-download SAM3.

## Architecture

The system turns webcam frames into legal chess moves through a five-stage pipeline. Each stage
lives in one module under `src/chess_detection/`:

1. **Detection** (`detection.py`) — SAM3 with the text prompt `"chess pieces"` returns boxes,
   scores, and masks; `from_sam` converts them into an `sv.Detections`. Piece *identity and color
   are never recovered* — only "something is here".
2. **Calibration** (`calibration.py`) — the user clicks the board's 4 corners. `build_dst_grid`
   fits a perspective transform from a canonical 9×9 lattice to those corners, and `grid_centers`
   averages each cell's 4 vertices into 64 square centers.
3. **Matching** (`tracking.py`) — each detection's *bottom-middle* point (where the piece meets
   the board, not its centroid) is greedily assigned to the nearest unused square center within
   `BBOX_MATCH_DISTANCE_THRESHOLD`. Output is a 64-element occupancy list of bools.
4. **Filtering** (`stability.py`) — `stabilize` majority-votes each square over a sliding window
   and then waits for the voted reading to hold still, so neither detector jitter nor a hand
   crossing the board reaches move inference.
5. **Move inference** (`board.py`) — `infer_move` compares the change against what every legal
   move would do, and plays the one that fits.

Stages 1-3 run per frame and are noisy; 4 is what makes 5 trustworthy. When "the board does not
update", the fault is almost always in 2, 3 or 4 — the chess logic in 5 is covered by
`tests/test_inference.py`.

### The index contract

The occupancy list index *is* the `python-chess` square number (a1=0 … h8=63), so `positions[i]`
can be compared directly with `board.piece_at(i)`. Nothing checks this. Get it wrong and every
reading is still internally consistent — it just lands on the wrong squares, so no change is ever
a legal move and the board silently never advances.

**Corner-click order is what fixes it.** The corners are clicked `a1 → h1 → h8 → a8`, going
around the board (`CORNER_ORDER` in `calibration.py`, and the UI names each corner as it asks for
it). Clicking *screen* corners instead — top-left, top-right, bottom-right, bottom-left — lands on
a mirrored mapping whenever the camera looks from Black's side, which is the normal way to set this
up. A mirror is not a rotation: four rotations cover only four of the eight ways a grid can sit on
a board, so no `BOARD_ROTATION` value can undo one. `tests/test_mapping.py` plays a real game
through both orders and pins the difference down:

```
physical: e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1
a1-first: e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1
screen:   d2d4 d7d5 b1c3 g8f6 c1f4 b8c6 MISS(Change does not match any legal move)
```

Every move comes back as its mirror image, and castling stops resolving because the king and queen
have swapped squares. Most non-royal moves still look plausible, which is what makes it nasty.

`BOARD_ROTATION` (default 0) remains as the escape hatch for a click order that is rotated rather
than mirrored. `build_dst_grid` applies it as `np.rot90(dst_grid, k=rotation)`; the whole effect on
the occupancy list is a pure index permutation, which is why `squares.py` holds it
(`rotation_permutation`, `reinterpret`) with no geometry and no opencv: rotating the 9×9 lattice
and then reading off cells is the same as reading off cells and rotating the 8×8 cell array.
`reinterpret` re-reads one detection under a different rotation without re-running the detector,
which is what makes `rotation_hint` possible.

On the fourth click, `detect_rotation` scores all four rotations against a starting position and
reports the match ("63/64"). Treat that as a calibration check rather than a fix: it cannot see a
mirror, because a start position is unchanged by one. Rotations 0 and 2 always tie for the same
reason, so the tie is broken toward whatever is already selected.

To verify rather than guess, leave "Label squares on the live view" on: `annotate_squares` writes
each square's name at its center, green when occupied. Set up a known position and read it off the
live frame — a1 must sit where a1 physically is.

`BOARD_GRID_SIZE = 9` counts grid *lines*, not squares (9 lines → 8×8 = 64 cells).

### Move inference is occupancy-only, and diffs two *detections*

`infer_move` does not pattern-match the shape of the occupancy delta. That needs a branch per
move type and quietly gets captures wrong — a capture changes the occupancy of exactly *one*
square (the origin empties; the destination was already occupied), which looks nothing like a
quiet move's two. Every legal move is pushed instead, and the squares it empties and fills are
compared against the observed change.

The comparison is against the **previous detection** (`baseline`), not against the board's own
occupancy. Absolute matching only works if the detector sees all 32 pieces, and it does not: a
piece behind another is missed on every frame, so an absolute comparison disagrees forever and no
move is ever found. Diffing two detections cancels any bias steady between them. The first settled
reading after calibration therefore *establishes* the baseline and plays nothing — it records how
the camera sees the board as it stands, blind spots included.

`baseline` only advances when a reading was understood (a move, or no change). An unmatched frame
leaves it alone, so a later cleaner reading still diffs against the last position that made sense.

`match_cost` decides whether a move explains an observation. Fills must match exactly. A square
the move empties but that was never observed emptying is forgiven **only if that square was
already absent from the baseline** — i.e. the detector could not see a piece there to begin with.
That one test separates the two readings that otherwise look identical:

| observation | verdict |
|---|---|
| f6 filled, g8 never visible | `g8f6` — the hidden knight moved |
| e4 filled, e2 still plainly visible | rejected — nothing left e2, so this is noise |

`MOVE_MATCH_TOLERANCE` (default 2, enough for castling's king *and* rook) caps how many squares
may be hidden this way. Exact matches always win, so it only decides what happens when nothing
matches exactly. Ties across different from/to squares are reported as ambiguous and dropped
rather than guessed; ties among promotions share a from/to and resolve to a queen.

`update_board` wraps this and returns `(board, move, message, baseline)`. The message is what the
UI status line shows, so "why didn't the board update" is answered on screen rather than on stdout.

### Occupancy is filtered before it is believed (`stability.py`)

Requiring N *identical* raw frames does not work: SAM3 adds or drops a piece often enough that 64
bools rarely repeat exactly, so nothing ever confirms. `stabilize` runs two filters instead — a
per-square majority vote over the last `VOTE_WINDOW` frames, and then a requirement that the voted
reading repeat `SETTLE_FRAMES` times before it is emitted. The second filter is what discards the
states a move passes *through*: a piece in the air has its origin empty and its destination not
yet filled, which is indistinguishable from a legal capture.

This lives apart from `tracking.py` on purpose — that module imports `detection`, and this one
depends on nothing but `config`, so it is testable without a GPU.

### Gradio wiring (`run.py`)

Six `gr.State` objects carry everything: `state_points` (the 4 clicked corners),
`state_positions` (the 64 bools), `state_board` (the `chess.Board`), `state_stability` (the vote
window and settle counter), `state_baseline` (the last understood detection) and `state_signal`.

**A component listed as a stream output is re-rendered on every tick.** `gr.skip()` does not
prevent this: it returns `{"__type__": "update"}`, which `blocks.py` treats as a prop update —
it reconstructs the block and sends it anyway, so the board SVG still flashes twice a second.
The fix is structural, not a flag. `cam.stream` outputs only the live image and the states; the
board, status line, move list and diagnostics hang off `state_signal.change` instead.

`state_signal` is a plain tuple `(moves, message, emptied, filled, hint)`, rebuilt only when a
settled reading is interpreted. Gradio tracks it with `utils.deep_hash` before and after the
handler (`get_state_ids_to_track`) and fires the listener only when the hash differs, so the
board is redrawn on moves and on nothing else. It has to be a plain tuple: `state_board` is
mutated in place by `board.push`, so hashing it is not a reliable change signal.

Counters that legitimately change every frame — raw detections, how full the vote window is, how
far the reading has settled — are drawn onto the video by `annotate_hud` rather than put in a
Markdown component. The frame is replaced every tick regardless, so text on it costs no extra
redraw.

When a settled change matches no legal move, `rotation_hint` re-reads that same detection under
the other three rotations and names the one that would have explained it ("rotation 3 would read
this as e4"). A wrong rotation is otherwise indistinguishable from a broken detector: every
reading is internally consistent, it just lands on the wrong squares.

Before 4 corners are clicked, `track_movements` returns an empty positions list and the board stays
frozen. "Calibrate" freezes the current frame and resets the points, stabilizer, baseline and
signal; corners are then collected by clicking the frozen image. "Undo move" and "New game" reset
the stabilizer and baseline too, so the next settled reading re-registers against the changed board.

### Model loading

`detection.py` builds the SAM3 model at *import time*, guarded by `if gr.NO_RELOAD`. The package
`__init__` is a lazy facade (PEP 562 `__getattr__`) precisely because of that: importing
`chess_detection` costs nothing, and only touching a name that lives in `tracking` or `detection`
pulls the weights onto the GPU. `config.py` is kept free of third-party imports for the same
reason, which is why the palette lives there as `COLOR_PALETTE_HEX` (plain strings) and is turned
into an `sv.ColorPalette` in `detection.py`.

Two thresholds are in play: the processor is constructed at 0.3, then detections are filtered
again at `DEFAULT_CONFIDENCE_THRESHOLD` (0.5) in `detect_chess_pieces`.

`Sam3Processor` does **not** enter an autocast context of its own — sam3's video predictors carry
a `bf16_context`, the image processor does not, and its own example notebook enters a global
`torch.autocast("cuda", dtype=torch.bfloat16)` before using it. Parts of the ViTDet backbone emit
bfloat16 activations regardless, so calling `set_image` outside autocast dies with
`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`. That is why
`detect_chess_pieces` wraps both `set_image` and `set_text_prompt` in `torch.autocast`, and
`02_piece_detection.ipynb` carries the same setup in a cell of its own.

### Notebooks

`notebooks/02_piece_detection.ipynb` is the prototype the detection/annotation code in
`detection.py` was lifted from; it runs against `assets/sample_board.jpg` and is the fastest way
to test detection without a webcam. `notebooks/webcam_shot.py` grabs a single frame using the
Windows `CAP_DSHOW` backend (host-side — the container has no camera device).
