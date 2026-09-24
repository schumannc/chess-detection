from typing import Iterable, Optional

import chess
import chess.svg

from .config import (
    BOARD_SVG_SIZE,
    MOVE_MATCH_TOLERANCE,
)


# Which squares hold a piece at the start of a game. Occupancy alone cannot tell
# White from Black, but it does pin down which two edges of the grid the pieces
# sit on, which is what fixes the rotation.
START_OCCUPANCY = {
    square for square in range(64) if chess.square_rank(square) in (0, 1, 6, 7)
}


def board_occupancy(board: chess.Board) -> set[int]:
    """Squares currently holding a piece, as python-chess square numbers."""
    return set(chess.SquareSet(board.occupied))


def occupancy_from_positions(positions: Iterable[bool]) -> set[int]:
    """Convert the 64-element occupancy list into a set of square numbers.

    `positions[i]` is the occupancy of python-chess square `i` (a1=0 ... h8=63);
    see "the index contract" in CLAUDE.md.
    """
    return {i for i, has_piece in enumerate(positions) if has_piece}


def positions_from_occupancy(squares) -> list[bool]:
    """Inverse of `occupancy_from_positions`."""
    return [square in squares for square in range(64)]


def occupancy_after(board: chess.Board, move: chess.Move) -> set[int]:
    """Occupancy the board *would* have if `move` were played."""
    board.push(move)
    try:
        return board_occupancy(board)
    finally:
        board.pop()


def move_delta(board: chess.Board, move: chess.Move) -> tuple[set[int], set[int]]:
    """Squares `move` empties and squares it fills, as (emptied, filled)."""
    before = board_occupancy(board)
    after = occupancy_after(board, move)
    return before - after, after - before


def match_cost(
    reference: set[int],
    emptied: set[int],
    filled: set[int],
    move_emptied: set[int],
    move_filled: set[int],
) -> Optional[int]:
    """How well a move explains an observation, or None if it cannot.

    A square the move empties but that the detector never reported emptying is
    only forgivable when the detector could not see a piece there in the first
    place (`s not in reference`) — a piece hidden behind another is missed on
    every frame, so it is also missed when it leaves.

    That test is what separates a hidden piece from a phantom one. Both look like
    "a square filled, with no origin observed", but if the origin is plainly
    visible in the reference and stays visible, the piece did not move and the
    reading is noise, not a move.
    """
    if filled != move_filled:
        return None
    if emptied - move_emptied:
        return None
    unseen = move_emptied - emptied
    if any(square in reference for square in unseen):
        return None
    return len(unseen)


def infer_move(
    board: chess.Board,
    detected: set[int],
    baseline: Optional[set[int]] = None,
    tolerance: int = MOVE_MATCH_TOLERANCE,
) -> tuple[Optional[chess.Move], str]:
    """Find the legal move that explains how `detected` differs from `baseline`.

    The comparison is against the previous *detection*, not against the board's
    own occupancy. Matching absolute positions only works if the detector sees all
    32 pieces, and it does not — a piece behind another is missed on every frame,
    so an absolute comparison disagrees forever and no move is ever found. Diffing
    two detections cancels any bias steady between them, leaving the squares that
    actually changed.

    `tolerance` is how many squares a move may empty without the detector noticing
    (see `match_cost`). Exact matches always win, so it only decides what happens
    when nothing matches exactly.
    """
    reference = board_occupancy(board) if baseline is None else set(baseline)
    emptied = reference - detected
    filled = detected - reference

    if not emptied and not filled:
        return None, "no-change"
    if board.is_game_over():
        return None, "game-over"

    scored = []
    for move in board.legal_moves:
        move_emptied, move_filled = move_delta(board, move)
        cost = match_cost(reference, emptied, filled, move_emptied, move_filled)
        if cost is not None and cost <= tolerance:
            scored.append((cost, move))

    if not scored:
        return None, "unmatched"

    best = min(cost for cost, _ in scored)
    tied = [move for cost, move in scored if cost == best]

    # Promotions to different pieces move the same piece to the same square, so a
    # tie among them is not real ambiguity — default to a queen.
    if len({(move.from_square, move.to_square) for move in tied}) > 1:
        return None, "ambiguous"
    for move in tied:
        if move.promotion == chess.QUEEN:
            return move, "ok"
    return tied[0], "ok"


def render_board(board: Optional[chess.Board] = None, flipped: bool = False) -> str:
    if board is None:
        board = chess.Board()
    lastmove = board.move_stack[-1] if board.move_stack else None
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board=board,
        size=BOARD_SVG_SIZE,
        lastmove=lastmove,
        check=check,
        orientation=chess.BLACK if flipped else chess.WHITE,
    )


def move_history(board: chess.Board) -> str:
    """The game so far in SAN, as markdown."""
    if not board.move_stack:
        return "_No moves yet._"
    try:
        return chess.Board().variation_san(board.move_stack)
    except ValueError:
        # Replay failed (board was edited out from under us) — fall back to UCI.
        return " ".join(move.uci() for move in board.move_stack)


def game_status(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate — {winner} wins."
    if board.is_stalemate():
        return "Stalemate."
    if board.is_insufficient_material():
        return "Draw — insufficient material."
    if board.can_claim_draw():
        return "Draw can be claimed."
    turn = "White" if board.turn == chess.WHITE else "Black"
    return f"{turn} to move" + (" — check!" if board.is_check() else ".")


REASON_MESSAGES = {
    "no-change": "Board unchanged.",
    "game-over": "Game is over — no further moves accepted.",
    "unmatched": (
        "Change does not match any legal move — check the grid rotation, "
        "or wait for the detection to settle."
    ),
    "ambiguous": "Detected position matches more than one legal move — ignored.",
}


def update_board(
    positions: list[bool],
    board: chess.Board,
    baseline: Optional[set[int]] = None,
    tolerance: int = MOVE_MATCH_TOLERANCE,
) -> tuple[chess.Board, Optional[chess.Move], str, Optional[set[int]]]:
    """Apply the move implied by `positions` to `board` (mutated in place).

    Returns the board, the move played (None if none was), a human-readable
    message, and the baseline to diff the *next* reading against. The baseline
    only advances when the reading was understood: an unmatched frame leaves it
    alone, so a later, cleaner reading can still be diffed against the last
    position that made sense.
    """
    if not positions:
        return board, None, "Waiting for calibration.", baseline

    detected = occupancy_from_positions(positions)

    if baseline is None:
        # First settled reading after calibration. It defines how the camera sees
        # the board as it stands — including the pieces it cannot see — and is not
        # a move. Comparing it against the board's own occupancy would fail for
        # exactly the pieces the detector is blind to.
        return board, None, "Baseline captured - make a move.", detected
    move, reason = infer_move(board, detected, baseline=baseline, tolerance=tolerance)

    if move is None:
        advanced = detected if reason == "no-change" else baseline
        return board, None, REASON_MESSAGES.get(reason, reason), advanced

    san = board.san(move)
    board.push(move)
    return board, move, f"Played {san}", detected
