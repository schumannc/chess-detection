"""Occupancy -> move inference, checked without a camera or a GPU.

Run it with `python tests/test_inference.py` (needs only `python-chess`); it exits
non-zero on failure. The interesting cases are not the clean ones — they are the
detector being blind to a piece, jittering between frames, or catching a board
mid-move, because that is what the pipeline actually sees.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import chess

from chess_detection.board import board_occupancy, infer_move, update_board
from chess_detection.board import occupancy_from_positions, positions_from_occupancy
from chess_detection.squares import reinterpret
from chess_detection.stability import new_stability_state, stabilize


def occ(b, uci=None):
    b = chess.Board(b.fen())
    if uci:
        b.push_uci(uci)
    return board_occupancy(b)


def as_list(s):
    return [i in s for i in range(64)]


fails = 0

print("--- every move type, clean detector ---")
cases = [
    ("quiet", chess.Board(), "e2e4"),
    ("capture", chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"), "e4d5"),
    ("O-O", chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"), "e1g1"),
    ("O-O-O", chess.Board("r3kbnr/pppqpppp/2np4/8/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5"), "e1c1"),
    ("en passant", chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"), "e5f6"),
    ("promotion", chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1"), "a7a8q"),
]
for name, b, uci in cases:
    mv, reason = infer_move(b, occ(b, uci))
    ok = mv == chess.Move.from_uci(uci)
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name:11s} {uci:6s} -> {mv} ({reason})")

print("\n--- detector permanently blind to the pieces on g8/h8 ---")
b = chess.Board()
BLIND = {chess.H8, chess.G8}
mv_abs, r_abs = infer_move(b, occ(b, "e2e4") - BLIND, baseline=None)
mv_rel, r_rel = infer_move(b, occ(b, "e2e4") - BLIND, baseline=occ(b) - BLIND)
print(f"  absolute match (old behaviour): {mv_abs} ({r_abs})")
print(f"  vs previous detection (new):    {mv_rel} ({r_rel})")
fails += mv_rel != chess.Move.from_uci("e2e4")

print("\n--- 10-move game through that blind detector ---")
b = chess.Board()
b, mv, msg, baseline = update_board(as_list(occ(b) - BLIND), b, None)
print("  first reading:", msg, "| move:", mv)
if mv is not None:
    fails += 1
for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f6e4", "d2d4", "e5d4"]:
    b, mv, msg, baseline = update_board(as_list(occ(b, uci) - BLIND), b, baseline)
    if mv is None:
        print("  FAIL at", uci, "->", msg)
        fails += 1
        break
print("  played:", " ".join(m.uci() for m in b.move_stack))

print("\n--- jitter: no two raw frames are ever identical ---")
st = new_stability_state()
truth = as_list(occ(chess.Board()))
emitted = None
for f in range(12):
    noisy = list(truth)
    noisy[20 + (f % 3)] = f % 2 == 0
    st, out = stabilize(st, noisy)
    if out and emitted is None:
        emitted = (f, out)
print("  first emit at frame:", emitted[0] if emitted else None,
      "| equals truth:", bool(emitted and emitted[1] == truth))
if not emitted or emitted[1] != truth:
    fails += 1

print("\n--- piece in the air must NOT be read as a capture ---")
b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
st = new_stability_state()
baseline = occ(b)
played = []
frames = [occ(b)] * 4 + [occ(b) - {chess.E4}] * 4 + [occ(b, "e4d5")] * 8
for reading in frames:
    st, out = stabilize(st, as_list(reading))
    if out:
        b, mv, msg, baseline = update_board(out, b, baseline)
        if mv:
            played.append(mv.uci())
print("  moves played:", played, "(want ['e4d5'])")
if played != ["e4d5"]:
    fails += 1

print("\n--- a hand over the board must play nothing ---")
b = chess.Board()
st = new_stability_state()
baseline = occ(b)
played = []
frames = [occ(b)] * 4 + [set(list(occ(b))[:12])] * 6 + [occ(b)] * 4 + [occ(b, "e2e4")] * 8
for reading in frames:
    st, out = stabilize(st, as_list(reading))
    if out:
        b, mv, msg, baseline = update_board(out, b, baseline)
        if mv:
            played.append(mv.uci())
print("  moves played:", played, "(want ['e2e4'])")
if played != ["e2e4"]:
    fails += 1

print("\n--- phantom vs hidden: identical delta, opposite verdicts ---")
b = chess.Board()
seen = occ(b)
# (a) something spurious appears on e4 while the pawn on e2 is plainly still there.
mv, reason = infer_move(b, seen | {chess.E4}, baseline=seen)
print(f"  phantom fill on e4, e2 still visible -> {mv} ({reason})  want None")
fails += mv is not None
# (b) same shape of delta, but the detector never saw the g8 knight at all.
#     (Black to move, so g8f6 is actually legal.)
bb = chess.Board()
bb.push_uci("e2e4")
blind = board_occupancy(bb) - {chess.G8}
mv, reason = infer_move(bb, blind | {chess.F6}, baseline=blind)
print(f"  hidden g8 knight lands on f6         -> {mv} ({reason})  want g8f6")
fails += mv != chess.Move.from_uci("g8f6")

print("\n--- ambiguity is refused, not guessed ---")
# Kings off the back rank so both rooks really can reach d1.
b = chess.Board("8/4k3/8/8/8/8/4K3/R6R w - - 0 1")
blind = board_occupancy(b) - {chess.A1, chess.H1}
mv, reason = infer_move(b, blind | {chess.D1}, baseline=blind)
print(f"  d1 filled, both rooks invisible      -> {mv} ({reason})  want None")
fails += mv is not None

print("\n--- a wrong grid rotation is named, not just rejected ---")
# The camera is physically aligned for rotation 3, but the app is set to 1.
TRUE_ROT, ACTIVE_ROT = 3, 1
b = chess.Board()
seen_before = reinterpret(positions_from_occupancy(occ(b)), TRUE_ROT, ACTIVE_ROT)
seen_after = reinterpret(positions_from_occupancy(occ(b, "e2e4")), TRUE_ROT, ACTIVE_ROT)
mv, reason = infer_move(b, occupancy_from_positions(seen_after),
                        baseline=occupancy_from_positions(seen_before))
print(f"  read under the wrong rotation {ACTIVE_ROT} -> {mv} ({reason})  want None")
fails += mv is not None

named = []
for k in range(4):
    if k == ACTIVE_ROT:
        continue
    m, _ = infer_move(
        b,
        occupancy_from_positions(reinterpret(seen_after, ACTIVE_ROT, k)),
        baseline=occupancy_from_positions(reinterpret(seen_before, ACTIVE_ROT, k)),
    )
    if m is not None:
        named.append((k, b.san(m)))
print(f"  rotations that would explain it: {named}  want [({TRUE_ROT}, 'e4')]")
if named != [(TRUE_ROT, "e4")]:
    fails += 1

print("\nFAILS:", fails)
sys.exit(1 if fails else 0)
