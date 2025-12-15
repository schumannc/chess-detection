from supervision import ColorPalette


# Model configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.45
DETECTION_PROMPT = "chess pieces"

# Visualization
COLOR_PALETTE = ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# Tracking
BBOX_MATCH_DISTANCE_THRESHOLD = 50.0

# Board calibration
BOARD_GRID_SIZE = 9