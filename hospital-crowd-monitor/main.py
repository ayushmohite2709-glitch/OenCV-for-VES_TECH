"""
============================================================
REAL-TIME HOSPITAL CROWD MONITORING SYSTEM
Live webcam -> YOLO11n person detection -> ByteTrack -> virtual
counting line -> Entry/Exit event engine -> occupancy vs.
registered-visitor comparison.
============================================================
"""

import sys
import time
from dataclasses import dataclass, field
from collections import deque

import cv2

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO


# ============================================================
# CONFIGURATION  (left unchanged from the working baseline —
# see the accompanying explanation for why each value is kept)
# ============================================================

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0                 # MUST KEEP: laptop built-in webcam

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

LINE_POSITION = 0.50             # virtual counting line at 50% of width
COUNTING_ZONE = 80                # +/- tolerance zone around the line (px)

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50

TRACK_TIMEOUT = 60                # frames a lost ByteTrack ID is retained

POSITION_HISTORY = 5              # moving-average smoothing window
MIN_MOVEMENT = 12                 # minimum px displacement to accept a crossing
OPPOSITE_SIDE_CONFIRM_FRAMES = 4  # consecutive frames required on the far side
CROSSING_COOLDOWN = 15            # min frames between two counts, same track

MAX_CONSECUTIVE_READ_FAILURES = 30  # webcam glitch tolerance before giving up


# ============================================================
# VISUAL / LABEL CONSTANTS
# ============================================================

# Colors are BGR (OpenCV convention)
COLOR_CORRIDOR = (255, 170, 40)      # blue-ish  -> CORRIDOR / OUTSIDE
COLOR_ROOM = (40, 170, 255)          # orange-ish -> ROOM / INSIDE
COLOR_LINE = (255, 0, 0)
COLOR_ZONE_BOUNDARY = (100, 100, 255)
COLOR_ENTRY = (0, 220, 0)
COLOR_EXIT = (0, 140, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DASH_BG = (30, 30, 30)

SIDE_LABELS = {
    "CORRIDOR": "CORRIDOR / OUTSIDE",
    "ROOM": "ROOM / INSIDE",
    "ZONE": "COUNTING ZONE",
    "UNKNOWN": "DETECTING...",
}


def physical_label(side):
    return SIDE_LABELS.get(side, side)


# ============================================================
# REGISTRATION INPUT
# (FIX: retries on bad input instead of exiting the program)
# ============================================================

def ask_registered_visitors():
    while True:
        raw = input("Enter total number of registered visitors: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("ERROR: Please enter a valid whole number.")
            continue
        if value < 0:
            print("ERROR: Please enter a positive number.")
            continue
        return value


total_registered = ask_registered_visitors()


# ============================================================
# LOAD YOLO MODEL
# (FIX: wrapped in try/except so a missing/corrupt model file
#  exits cleanly instead of an unhandled traceback)
# ============================================================

print()
print("Loading YOLO model...")

try:
    model = YOLO(MODEL_PATH)
except Exception as exc:
    print(f"ERROR: Could not load YOLO model '{MODEL_PATH}': {exc}")
    sys.exit(1)

# ENHANCEMENT: use the RTX 3050 automatically if CUDA is available.
# A stable higher frame rate directly improves crossing accuracy because
# the position-smoothing / consecutive-frame confirmation logic gets more
# temporal samples per second of real movement.
DEVICE = "cpu"
if torch is not None and torch.cuda.is_available():
    DEVICE = 0
    print(f"YOLO model loaded. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("YOLO model loaded. Using CPU (no CUDA GPU detected).")


# ============================================================
# OPEN WEBCAM
# (FIX: falls back to the default backend if CAP_DSHOW fails to
#  open on this machine, instead of hard-exiting)
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Hospital Crowd Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hospital Crowd Monitor", FRAME_WIDTH, FRAME_HEIGHT)


# ============================================================
# GLOBAL COUNTERS
# (FIX: present/not_in_room initialized BEFORE the loop so the
#  final summary never raises NameError if the loop exits early)
# ============================================================

entry_count = 0
exit_count = 0
present = 0
not_in_room = total_registered


# ============================================================
# PERSON TRACK STRUCTURE
# ============================================================

@dataclass
class PersonTrack:
    # Side the person was confirmed on BEFORE entering the counting zone.
    # "CORRIDOR" | "ROOM" | "UNKNOWN"
    confirmed_side: str = "UNKNOWN"

    # Side recorded at the moment the person entered the counting zone
    zone_entry_side: str = None

    # Smoothed x-position recorded at the moment of zone entry
    zone_entry_x: int = None

    # Smoothed X position history (moving average window)
    x_history: deque = field(default_factory=lambda: deque(maxlen=POSITION_HISTORY))

    # Consecutive frames confirmed on the opposite side, with valid movement
    opposite_side_frames: int = 0

    last_seen: int = 0
    last_crossing_frame: int = -999999

    # False until the person's initial side has been established.
    # A person first seen inside the zone stays un-armed -> never counted
    # for that first, ambiguous appearance.
    armed: bool = False


person_tracks = {}
frame_number = 0


# ============================================================
# HELPER: GET SIDE
# ============================================================

def get_side(center_x, line_x):
    """Classifies a smoothed x-position as CORRIDOR, ROOM, or ZONE."""
    zone_left = line_x - COUNTING_ZONE
    zone_right = line_x + COUNTING_ZONE

    if center_x < zone_left:
        return "CORRIDOR"
    elif center_x > zone_right:
        return "ROOM"
    else:
        return "ZONE"


# ============================================================
# HELPER: SMOOTH X POSITION
# ============================================================

def smooth_x(track, center_x):
    """Short moving average to reduce bounding-box jitter near the line."""
    track.x_history.append(center_x)
    return int(sum(track.x_history) / len(track.x_history))


# ============================================================
# CROSSING PROCESSOR — per-track state machine
#
# FIX #1 (critical): the original code had Entry/Exit REVERSED.
#   Old code counted RIGHT->ZONE->LEFT as ENTRY and LEFT->ZONE->RIGHT
#   as EXIT. Per the project spec, LEFT = CORRIDOR/OUTSIDE and
#   RIGHT = ROOM/INSIDE, so the correct mapping is:
#     CORRIDOR -> ZONE -> ROOM  = ENTRY
#     ROOM -> ZONE -> CORRIDOR  = EXIT
#   This version implements that mapping directly with semantic
#   side names instead of generic LEFT/RIGHT.
#
# FIX #2: the original code could return "ENTRY"/"EXIT" (and show
#   "...CONFIRMED" on screen) even when the cooldown blocked the
#   actual counter increment. This version only returns an event
#   string when the corresponding counter was actually incremented.
#
# ENHANCEMENT: movement validation now checks DIRECTION, not just
#   magnitude — a crossing must move the correct way (rightward for
#   an entry, leftward for an exit), not merely move some minimum
#   distance from the zone-entry point.
# ============================================================

def process_crossing(track, current_side, current_x, current_frame):
    """
    Valid ENTRY : CORRIDOR / OUTSIDE -> COUNTING ZONE -> ROOM / INSIDE
    Valid EXIT  : ROOM / INSIDE      -> COUNTING ZONE -> CORRIDOR / OUTSIDE

    Returns "ENTRY", "EXIT", or None.
    """
    global entry_count, exit_count

    # ---- First valid observation of this track ----
    if track.confirmed_side == "UNKNOWN":
        if current_side in ("CORRIDOR", "ROOM"):
            track.confirmed_side = current_side
            track.armed = True
        # First seen inside the ZONE: stay UNKNOWN/un-armed. Do NOT count.
        return None

    # ---- Currently inside the counting zone ----
    if current_side == "ZONE":
        if track.zone_entry_side is None:
            track.zone_entry_side = track.confirmed_side
            track.zone_entry_x = current_x
            track.opposite_side_frames = 0
        # confirmed_side is intentionally left untouched while in the zone.
        return None

    # ---- Person is now on CORRIDOR or ROOM ----
    came_from_zone = track.zone_entry_side is not None
    opposite_side = current_side != track.confirmed_side

    if not came_from_zone or not opposite_side:
        # Either the track jumped sides without registering a zone dwell
        # (large frame-to-frame jump), or the person entered the zone and
        # returned to the SAME side they came from. Neither is a crossing.
        track.confirmed_side = current_side
        track.zone_entry_side = None
        track.zone_entry_x = None
        track.opposite_side_frames = 0
        return None

    # A genuine opposite-side transition through the zone is in progress.
    movement = current_x - track.zone_entry_x

    if track.confirmed_side == "CORRIDOR":
        direction_ok = movement >= MIN_MOVEMENT     # must move rightward
        event_type = "ENTRY"
    else:  # confirmed_side == "ROOM"
        direction_ok = movement <= -MIN_MOVEMENT     # must move leftward
        event_type = "EXIT"

    if direction_ok and track.armed:
        track.opposite_side_frames += 1
    else:
        track.opposite_side_frames = 0

    event = None

    if track.opposite_side_frames >= OPPOSITE_SIDE_CONFIRM_FRAMES:
        if current_frame - track.last_crossing_frame >= CROSSING_COOLDOWN:
            if event_type == "ENTRY":
                entry_count += 1
            else:
                exit_count += 1
            track.last_crossing_frame = current_frame
            event = event_type
            print(f"[{event_type}] Person crossing confirmed | "
                  f"Entries={entry_count} | Exits={exit_count}")
        # Whether or not the cooldown blocked the count, the person has
        # physically finished the crossing -> reset state either way so
        # the track doesn't get stuck waiting on a phantom zone dwell.
        track.confirmed_side = current_side
        track.zone_entry_side = None
        track.zone_entry_x = None
        track.opposite_side_frames = 0

    return event


# ============================================================
# STARTUP MESSAGE
# (FIX: direction messages corrected to match the actual logic)
# ============================================================

print()
print("==============================================")
print("       HOSPITAL CROWD MONITOR")
print("==============================================")
print(f"REGISTERED VISITORS : {total_registered}")
print()
print("CORRIDOR / OUTSIDE -> ROOM / INSIDE : ENTRY")
print("ROOM / INSIDE -> CORRIDOR / OUTSIDE : EXIT")
print()
print(f"COUNTING ZONE : +/- {COUNTING_ZONE} pixels")
print(f"POSITION SMOOTHING : {POSITION_HISTORY} frames")
print(f"CROSSING CONFIRMATION : {OPPOSITE_SIDE_CONFIRM_FRAMES} frames")
print()
print("Press Q to quit.")
print("==============================================")
print()


# ============================================================
# MAIN LIVE CAMERA LOOP
# (FIX: wrapped in try/finally so the camera and windows are
#  always released cleanly, even on an unexpected error)
# ============================================================

consecutive_read_failures = 0
prev_time = time.time()
fps = 0.0

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            consecutive_read_failures += 1
            print(f"WARNING: Could not read frame "
                  f"({consecutive_read_failures}/{MAX_CONSECUTIVE_READ_FAILURES}).")
            if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print("ERROR: Webcam stopped responding. Exiting.")
                break
            continue
        consecutive_read_failures = 0

        frame_number += 1

        # MUST KEEP EXACTLY AS-IS: mirror the frame.
        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT),
                            interpolation=cv2.INTER_AREA)

        height, width = frame.shape[:2]

        line_x = int(width * LINE_POSITION)
        zone_left = line_x - COUNTING_ZONE
        zone_right = line_x + COUNTING_ZONE

        # --------------------------------------------------
        # ENHANCEMENT: light tinted overlay for each half, so
        # it's immediately obvious at a glance which side is
        # the CORRIDOR/OUTSIDE (entry source) and which is the
        # ROOM/INSIDE (exit source), without obscuring the feed.
        # --------------------------------------------------
        tint = frame.copy()
        cv2.rectangle(tint, (0, 0), (line_x, height), COLOR_CORRIDOR, -1)
        cv2.rectangle(tint, (line_x, 0), (width, height), COLOR_ROOM, -1)
        cv2.addWeighted(tint, 0.10, frame, 0.90, 0, frame)

        # Main counting line
        cv2.line(frame, (line_x, 0), (line_x, height), COLOR_LINE, 3)

        # Zone tolerance boundaries (visual only, not separate counting lines)
        cv2.line(frame, (zone_left, 0), (zone_left, height), COLOR_ZONE_BOUNDARY, 1)
        cv2.line(frame, (zone_right, 0), (zone_right, height), COLOR_ZONE_BOUNDARY, 1)

        cv2.putText(frame, "COUNTING LINE", (line_x - 105, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LINE, 2)

        # --------------------------------------------------
        # REQUIREMENT: label the two halves and the entry/exit
        # direction directly on the live feed.
        # --------------------------------------------------
        cv2.putText(frame, "CORRIDOR / OUTSIDE", (30, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CORRIDOR, 2)
        cv2.putText(frame, "ENTRY ->", (30, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ENTRY, 2)

        room_label_size = cv2.getTextSize("ROOM / INSIDE", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, "ROOM / INSIDE", (width - room_label_size[0] - 30, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ROOM, 2)
        exit_label_size = cv2.getTextSize("<- EXIT", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(frame, "<- EXIT", (width - exit_label_size[0] - 30, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_EXIT, 2)

        # --------------------------------------------------
        # YOLO + ByteTrack inference (person class only)
        # --------------------------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        for result in results:

            if result.boxes.id is None:
                continue

            boxes = result.boxes
            track_ids = result.boxes.id

            for box, track_id in zip(boxes, track_ids):

                person_id = int(track_id)

                if person_id not in person_tracks:
                    person_tracks[person_id] = PersonTrack()

                track = person_tracks[person_id]
                track.last_seen = frame_number

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                smoothed_x = smooth_x(track, center_x)
                current_side = get_side(smoothed_x, line_x)

                crossing_event = process_crossing(
                    track, current_side, smoothed_x, frame_number
                )

                # Individual bounding box (NOT a shared ROI rectangle)
                box_color = (
                    COLOR_CORRIDOR if track.confirmed_side == "CORRIDOR"
                    else COLOR_ROOM if track.confirmed_side == "ROOM"
                    else COLOR_WHITE
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.putText(frame, f"Person #{person_id} {confidence:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                side_text = physical_label(
                    current_side if current_side == "ZONE" else track.confirmed_side
                )
                cv2.putText(frame, side_text, (x1, min(y2 + 20, height - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

                if crossing_event == "ENTRY":
                    cv2.putText(frame, "ENTRY CONFIRMED", (x1, min(y2 + 42, height - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ENTRY, 2)
                elif crossing_event == "EXIT":
                    cv2.putText(frame, "EXIT CONFIRMED", (x1, min(y2 + 42, height - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_EXIT, 2)

        # --------------------------------------------------
        # Remove stale tracks
        # --------------------------------------------------
        expired_ids = [
            pid for pid, trk in person_tracks.items()
            if frame_number - trk.last_seen > TRACK_TIMEOUT
        ]
        for pid in expired_ids:
            person_tracks.pop(pid, None)

        # --------------------------------------------------
        # Occupancy (event-based, never from live box count)
        # --------------------------------------------------
        present = max(entry_count - exit_count, 0)
        not_in_room = max(total_registered - present, 0)
        extra_people = max(present - total_registered, 0)

        if present > total_registered:
            status = "OVER CAPACITY"
            alert_message = f"ALERT: {extra_people} EXTRA PERSON(S)"
            alert_type = "DANGER"
        elif present == total_registered:
            status = "ALL VISITORS PRESENT"
            alert_message = "REGISTERED COUNT MATCHED"
            alert_type = "OK"
        else:
            remaining = total_registered - present
            status = "VISITORS STILL EXPECTED"
            alert_message = f"{remaining} REGISTERED VISITOR(S) NOT IN ROOM"
            alert_type = "WAITING"

        # --------------------------------------------------
        # Dashboard
        # --------------------------------------------------
        cv2.rectangle(frame, (20, 20), (385, 255), COLOR_DASH_BG, -1)
        cv2.rectangle(frame, (20, 20), (385, 255), COLOR_WHITE, 2)

        cv2.putText(frame, f"REGISTERED : {total_registered}", (40, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)
        cv2.putText(frame, f"ENTRIES    : {entry_count}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ENTRY, 2)
        cv2.putText(frame, f"EXITS      : {exit_count}", (40, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_EXIT, 2)
        cv2.putText(frame, f"PRESENT    : {present}", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)
        cv2.putText(frame, f"NOT IN ROOM: {not_in_room}", (40, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {status}", (40, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2)

        # Alert bar
        if alert_type == "DANGER":
            alert_bg = (0, 0, 180)
        elif alert_type == "WAITING":
            alert_bg = (0, 100, 180)
        else:
            alert_bg = (0, 130, 0)

        cv2.rectangle(frame, (405, 20), (900, 70), alert_bg, -1)
        cv2.putText(frame, alert_message, (420, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)

        # Config + FPS readout (bottom-right)
        cv2.putText(frame, f"ZONE +/- {COUNTING_ZONE}px", (width - 225, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 225, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hospital Crowd Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except Exception as exc:
    print(f"ERROR: Unexpected failure in main loop: {exc}")

finally:
    # ========================================================
    # CLEANUP — always runs, even on error or Ctrl+C
    # ========================================================
    cap.release()
    cv2.destroyAllWindows()

    print()
    print("==============================================")
    print("             SESSION SUMMARY")
    print("==============================================")
    print(f"Registered : {total_registered}")
    print(f"Entries    : {entry_count}")
    print(f"Exits      : {exit_count}")
    print(f"Present    : {present}")
    print(f"Not in room: {not_in_room}")
    print("==============================================")