"""Webcam frames to legal chess moves.

Imports are lazy on purpose. `detection` builds the SAM3 model onto the GPU at
import time, so an eager re-export here would mean that merely naming this package
— in a test, a notebook or a one-off script — costs a model load. Resolving each
name on first access keeps the pure logic (`board`, `stability`) reachable with
nothing installed but `python-chess`.
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "START_OCCUPANCY": ".board",
    "board_occupancy": ".board",
    "game_status": ".board",
    "infer_move": ".board",
    "match_cost": ".board",
    "move_delta": ".board",
    "move_history": ".board",
    "occupancy_from_positions": ".board",
    "positions_from_occupancy": ".board",
    "render_board": ".board",
    "update_board": ".board",
    "CORNER_ORDER": ".calibration",
    "calibrate_board": ".calibration",
    "annotate_hud": ".calibration",
    "grid_centers": ".calibration",
    "render_calibration": ".calibration",
    "reinterpret": ".squares",
    "rotation_permutation": ".squares",
    "new_stability_state": ".stability",
    "stabilize": ".stability",
    "detect_rotation": ".tracking",
    "track_movements": ".tracking",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return __all__
