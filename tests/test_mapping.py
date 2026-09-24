"""Corner-click order, checked by playing a real game through the grid.

Needs opencv and supervision, so it runs in the container:

    docker compose run --rm app python tests/test_mapping.py

Pieces are placed at the image points where real squares are and read back through
the grid the app built. A test that hands the app square numbers directly cannot
see a mirrored mapping, because the mirror lives in the geometry.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import chess

from chess_detection.board import board_occupancy, update_board
from chess_detection.calibration import grid_centers
from chess_detection.tracking import match_positions

# Corners of the playing area in assets/sample_board.jpg. White sits at the far
# side of that board, which puts a1 at the far-right corner.
FAR_LEFT, FAR_RIGHT = [214.0, 206.0], [466.0, 206.0]
NEAR_LEFT, NEAR_RIGHT = [181.0, 363.0], [499.0, 363.0]

TRUTH = grid_centers([FAR_RIGHT, FAR_LEFT, NEAR_LEFT, NEAR_RIGHT], rotation=0)

GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]  # ends in O-O


def bboxes_for(squares):
    """One synthetic detection standing on each physical square."""
    return [
        [TRUTH[s][0] - 8, TRUTH[s][1] - 30, TRUTH[s][0] + 8, TRUTH[s][1]]
        for s in squares
    ]


def play_through(corners):
    """What the app reads while the given game is played on the real board."""
    centers = grid_centers(corners, rotation=0)
    physical, app, baseline, read = chess.Board(), chess.Board(), None, []

    app, _, _, baseline = update_board(
        match_positions(bboxes_for(board_occupancy(physical)), centers), app, baseline
    )
    for uci in GAME:
        physical.push_uci(uci)
        app, move, message, baseline = update_board(
            match_positions(bboxes_for(board_occupancy(physical)), centers), app, baseline
        )
        read.append(move.uci() if move else f"MISS({message})")
    return read


fails = 0

print("--- clicking a1 first reads the game faithfully ---")
read = play_through([FAR_RIGHT, FAR_LEFT, NEAR_LEFT, NEAR_RIGHT])
print("  physical:", " ".join(GAME))
print("  app read:", " ".join(read))
if read != GAME:
    print("  FAIL")
    fails += 1

print("\n--- clicking screen corners instead mirrors everything ---")
mirrored = play_through([FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT])
print("  app read:", " ".join(m[:24] for m in mirrored))
if mirrored == GAME:
    print("  FAIL: expected this order to be wrong, so the guard is not testing anything")
    fails += 1
elif mirrored[0] != "d2d4":
    print(f"  FAIL: expected e2e4 to be mirrored into d2d4, got {mirrored[0]}")
    fails += 1

print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
