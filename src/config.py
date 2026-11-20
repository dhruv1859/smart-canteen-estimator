"""
Central configuration for the Smart Canteen Crowd & Queue Estimator.
Edit ZONES to match your camera layout.
"""

# ── YOLO ────────────────────────────────────────────────────────────────────
YOLO_MODEL    = "yolov8n.pt"      # nano for speed; swap to yolov8s.pt for accuracy
YOLO_CONF     = 0.35              # detection confidence threshold
YOLO_CLASS_ID = 0                 # COCO class 0 = person

# ── Crowd thresholds ─────────────────────────────────────────────────────────
CROWD_THRESHOLDS = {
    "Low":      (0,  5),
    "Moderate": (6, 15),
    "High":     (16, 9999),
}

# ── Default zones (polygon vertices, pixel coords for a 640×480 frame) ───────
# Each zone: { "name": str, "color": (B,G,R), "points": [(x,y), ...] }
DEFAULT_ZONES = [
    {
        "name"  : "Seating Area",
        "color" : (0, 200, 0),        # green
        "points": [(10, 10), (300, 10), (300, 230), (10, 230)],
    },
    {
        "name"  : "Serving Counter",
        "color" : (0, 140, 255),      # orange
        "points": [(310, 10), (630, 10), (630, 230), (310, 230)],
    },
    {
        "name"  : "Billing Counter",
        "color" : (200, 0, 200),      # purple
        "points": [(10, 240), (300, 240), (300, 470), (10, 470)],
    },
    {
        "name"  : "Queue / Exit",
        "color" : (0, 0, 220),        # red
        "points": [(310, 240), (630, 240), (630, 470), (310, 470)],
    },
]

# ── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP_ALPHA   = 0.5     # blend weight for heatmap overlay
HEATMAP_DECAY   = 0.98    # frame-to-frame decay (1.0 = no decay)
HEATMAP_RADIUS  = 20      # Gaussian blob radius per detection

# ── Logging / CSV ─────────────────────────────────────────────────────────────
LOG_CSV_PATH    = "logs/occupancy_log.csv"
LOG_INTERVAL_S  = 5       # write a CSV row every N seconds

# ── Display ───────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FONT         = 0          # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.55
FONT_THICK   = 1

