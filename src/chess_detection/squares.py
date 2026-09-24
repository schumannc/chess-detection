"""The index contract: occupancy list index <-> python-chess square number.

Pure index arithmetic, deliberately free of the drawing and perspective code in
`calibration`, so the one part of the pipeline that fails *silently* can be tested
without opencv, supervision or a camera.
"""

import numpy as np


SQUARE_NAMES = [
    f"{file}{rank}"
    for rank in range(1, 9)
    for file in "abcdefgh"
]


def rotation_permutation(rotation: int) -> list[int]:
    """Square index under `rotation` -> the same physical cell under rotation 0.

    `build_dst_grid` rotates the 9x9 lattice of grid *lines*; rotating the lattice
    and then reading off cells is the same as reading off cells and rotating the
    8x8 cell array, so the whole effect of `rotation` on the occupancy list is this
    pure index permutation. No geometry needed.
    """
    return np.rot90(np.arange(64).reshape(8, 8), k=rotation).reshape(-1).tolist()


def reinterpret(positions: list[bool], from_rotation: int, to_rotation: int) -> list[bool]:
    """Re-read an occupancy list as if the grid had been rotated differently.

    Lets one detection be scored against all four rotations without re-running the
    detector, which is what turns "no legal move matches" into "rotation 3 reads
    this as e2e4".
    """
    source = rotation_permutation(from_rotation)
    inverse = [0] * 64
    for index, cell in enumerate(source):
        inverse[cell] = index
    return [positions[inverse[cell]] for cell in rotation_permutation(to_rotation)]
