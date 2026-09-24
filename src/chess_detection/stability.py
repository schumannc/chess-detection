"""Turning noisy per-frame occupancy into a reading worth acting on.

Kept apart from `tracking` on purpose: that module imports `detection`, which
builds the SAM3 model onto the GPU at import time. This one depends on nothing
but `config`, so the filtering can be exercised without any of that.
"""

from typing import Any, Optional

from .config import (
    SETTLE_FRAMES,
    VOTE_RATIO,
    VOTE_WINDOW,
)


def new_stability_state() -> dict[str, Any]:
    return {"window": [], "candidate": None, "count": 0, "confirmed": None}


def stabilize(
    state: Optional[dict[str, Any]],
    positions: list[bool],
    window_size: int = VOTE_WINDOW,
    ratio: float = VOTE_RATIO,
    settle: int = SETTLE_FRAMES,
) -> tuple[dict[str, Any], Optional[list[bool]]]:
    """Turn noisy per-frame occupancy into a reading worth acting on.

    Requiring N *identical* raw frames does not work in practice: SAM3 drops or
    adds a piece often enough that 64 bools rarely repeat exactly, so nothing ever
    confirms. Instead each square is majority-voted over a sliding window, and the
    voted reading must then repeat `settle` times before it is emitted.

    Returns the updated state and, only on the frame where a *new* reading settles,
    that reading. Every other frame yields None.
    """
    prev = state or new_stability_state()
    window = (prev["window"] + [list(positions)])[-window_size:]
    state = {
        "window": window,
        "candidate": prev["candidate"],
        "count": prev["count"],
        "confirmed": prev["confirmed"],
    }

    if len(window) < window_size:
        return state, None

    needed = max(1, round(len(window) * ratio))
    voted = [
        sum(frame[i] for frame in window) >= needed
        for i in range(len(positions))
    ]

    if voted == state["candidate"]:
        state["count"] += 1
    else:
        state["candidate"] = voted
        state["count"] = 1

    if state["count"] < settle or voted == state["confirmed"]:
        return state, None

    state["confirmed"] = voted
    return state, voted
