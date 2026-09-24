# Model configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DETECTION_PROMPT = "chess pieces"

# Visualization
COLOR_PALETTE_HEX = [
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
]

# Tracking
BBOX_MATCH_DISTANCE_THRESHOLD = 50.0

# Occupancy is read from a noisy detector, so a raw frame is never trusted on its
# own. Two filters run in series:
#   1. a per-square majority vote over the last VOTE_WINDOW frames, which absorbs
#      single-frame flicker (SAM3 dropping one piece for one frame);
#   2. that voted reading must then repeat SETTLE_FRAMES times before it is acted
#      on, which discards the transient states a move passes through — a piece in
#      the air has its origin empty and its destination not yet filled, and that
#      looks exactly like a legal capture.
VOTE_WINDOW = 5
VOTE_RATIO = 0.6
SETTLE_FRAMES = 3

# How many squares a move may empty without the detector noticing. This covers
# pieces the camera cannot see at all — a rook behind a knight is missed on every
# frame, including the frame it moves away on. Exact matches always win, so this
# only decides what happens when nothing matches exactly. Castling can hide two
# squares (king and rook), hence 2.
MOVE_MATCH_TOLERANCE = 2

# Board calibration
BOARD_GRID_SIZE = 9

# Quarter-turns applied to the calibrated grid before it is flattened into the
# 64-element occupancy list. This is what aligns `positions[i]` with python-chess
# square `i`; the right value depends on which corner was clicked first and where
# the camera sits. See "the index contract" in CLAUDE.md.
BOARD_ROTATION = 0

# Board rendering
BOARD_SVG_SIZE = 420
