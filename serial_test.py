import serial
import time
from ultralytics import YOLO
import cv2
from firebase_helper import publish_room_state, firebase_is_configured

# ── Serial Setup ──────────────────────────────────────────────────────────────
try:
    arduino = serial.Serial("COM7", 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    arduino_connected = True
    print("Arduino connected on COM7")
except Exception as e:
    arduino_connected = False
    print(f"Arduino not connected: {e} — running in simulation mode")


# ── Firebase Setup ────────────────────────────────────────────────────────────
firebase_enabled = firebase_is_configured()
if firebase_enabled:
    print("Firebase is configured — room state will be published")
else:
    print("Firebase not configured — running without cloud sync")


def send_command(cmd):
    if arduino_connected:
        arduino.write(f"{cmd}\n".encode())
        print(f"  [serial] Sent: {cmd}")


# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

# ── FSM State Constants ───────────────────────────────────────────────────────
IDLE = 0
LIGHTS = 1
FAN = 2
AIRCON = 3

STATE_NAMES = {IDLE: "IDLE", LIGHTS: "LIGHTS", FAN: "FAN", AIRCON: "AIRCON"}

# ── FSM State ─────────────────────────────────────────────────────────────────
current_state = IDLE

# ── Occupancy Tracking Settings ──────────────────────────────────────────────
OCCUPANCY_THRESHOLD = 5.0  # Seconds a person must be continuously detected
person_first_seen: dict[int, float] = {}  # {track_id: timestamp when first seen}
confirmed_persons: set[int] = set()  # Track IDs that passed the threshold
last_published_count: int | None = None  # Track Firebase publish changes

# ── OFF Delay Settings ────────────────────────────────────────────────────────
OFF_DELAY_SECONDS = 10.0  # How long to wait after desired state drops before acting

# ── FSM Hysteresis Buffers ───────────────────────────────────────────────────
off_timer_start = None    # When we started waiting to go to a lower/idle state

CONFIDENCE_THRESHOLD = 0.35


def count_to_target_state(count):
    """Map a stable person count to the desired FSM state."""
    if count >= 3:
        return AIRCON
    elif count >= 2:
        return FAN
    elif count >= 1:
        return LIGHTS
    else:
        return IDLE


def transition_to(new_state):
    """
    Transition the FSM from current_state to new_state.
    Always explicitly turns OFF devices from the old state
    before turning ON devices for the new state.
    """
    global current_state

    if new_state == current_state:
        return

    print(f"  [FSM] {STATE_NAMES[current_state]} → {STATE_NAMES[new_state]}")

    # ── Teardown: turn off what the OLD state was running ──────────────────
    if current_state == AIRCON:
        print("  [AC] OFF (display only)")
    if current_state >= FAN and new_state < FAN:
        send_command("FAN_OFF")
    if current_state >= LIGHTS and new_state < LIGHTS:
        send_command("LIGHT_OFF")

    # ── Setup: turn on what the NEW state requires ─────────────────────────
    if new_state >= LIGHTS and current_state < LIGHTS:
        send_command("LIGHT_ON")
    if new_state >= FAN and current_state < FAN:
        send_command("FAN_ON")
    if new_state == AIRCON:
        print("  [AC] ON (display only)")

    current_state = new_state


def get_debounced_count(raw_count):
    """
    Update occupancy tracking with time-based confirmation.
    Returns the confirmed occupant count.
    """
    current_time = time.time()
    
    # ── Collect IDs visible in this frame ────────────────────────────────────
    current_ids: set[int] = set()
    if len(raw_count) > 0:
        # raw_count is passed as a list of track_ids from the caller
        for track_id in raw_count:
            current_ids.add(track_id)
            if track_id not in person_first_seen:
                person_first_seen[track_id] = current_time
                print(f"  [tracker] Person ID {track_id} appeared – timer started.")

    # ── Reset timers for IDs that disappeared ────────────────────────────────
    lost_ids = set(person_first_seen.keys()) - current_ids
    for lost_id in lost_ids:
        elapsed = current_time - person_first_seen[lost_id]
        print(f"  [tracker] Person ID {lost_id} lost after {elapsed:.1f}s – timer reset.")
        del person_first_seen[lost_id]
        confirmed_persons.discard(lost_id)

    # ── Promote IDs that have been present long enough ───────────────────────
    for track_id in current_ids:
        elapsed = current_time - person_first_seen[track_id]
        if elapsed >= OCCUPANCY_THRESHOLD and track_id not in confirmed_persons:
            confirmed_persons.add(track_id)
            print(f"  [tracker] Person ID {track_id} CONFIRMED as occupant ({elapsed:.1f}s).")

    return len(confirmed_persons), len(current_ids)


# ─────────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()

    # Use YOLO tracker (ByteTrack) for persistent person IDs
    results = model.track(frame, persist=True, verbose=False, classes=[0])

    # ── Collect track IDs visible in this frame ───────────────────────────────
    track_ids = []
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().tolist()

    # ── Update occupancy tracking with time-based confirmation ────────────────
    confirmed_count, total_detected = get_debounced_count(track_ids)

    target_state = count_to_target_state(confirmed_count)

    # ── FSM Transition with OFF-Delay Hysteresis ──────────────────────────────
    if target_state > current_state:
        # Upgrading (more people): act immediately, reset off timer
        off_timer_start = None
        transition_to(target_state)

    elif target_state < current_state:
        # Downgrading (fewer people): start/check the off-delay timer
        if off_timer_start is None:
            off_timer_start = current_time
        elapsed = current_time - off_timer_start
        if elapsed >= OFF_DELAY_SECONDS:
            transition_to(target_state)
            off_timer_start = None

    else:
        # Same state: reset off timer (count stabilized back up)
        off_timer_start = None

    # ── Publish to Firebase (only when count changes) ─────────────────────────
    if firebase_enabled and confirmed_count != last_published_count:
        try:
            publish_room_state(confirmed_count, threshold=5)
            last_published_count = confirmed_count
        except Exception as e:
            print(f"  [Firebase] Error publishing state: {e}")

    # ── Draw Annotated Frame ──────────────────────────────────────────────────
    annotated_frame = results[0].plot()

    # ── HUD Overlay ───────────────────────────────────────────────────────────
    cv2.putText(
        annotated_frame,
        f"Detected: {total_detected}   Confirmed: {confirmed_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
    )

    cv2.putText(
        annotated_frame,
        f"State: {STATE_NAMES[current_state]}   Target: {STATE_NAMES[count_to_target_state(confirmed_count)]}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 0),
        2,
    )

    # Draw per-person timer progress bars
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id.int().tolist()):
            elapsed = current_time - person_first_seen.get(track_id, current_time)
            progress = min(elapsed / OCCUPANCY_THRESHOLD, 1.0)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            bar_total = x2 - x1
            bar_filled = int(bar_total * progress)
            # Orange while waiting, green once confirmed
            bar_color = (0, 255, 0) if track_id in confirmed_persons else (0, 165, 255)

            # Background track
            cv2.rectangle(annotated_frame, (x1, y2 + 2), (x2, y2 + 14), (50, 50, 50), -1)
            # Filled portion
            cv2.rectangle(annotated_frame, (x1, y2 + 2), (x1 + bar_filled, y2 + 14), bar_color, -1)

            # Status label
            status = "CONFIRMED" if track_id in confirmed_persons else f"{elapsed:.1f}s / {OCCUPANCY_THRESHOLD:.0f}s"
            cv2.putText(annotated_frame, status, (x1, y2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    hud_y = 115
    if current_state >= LIGHTS:
        cv2.putText(annotated_frame, "Light ON", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        hud_y += 40
    if current_state >= FAN:
        cv2.putText(annotated_frame, "Fan ON", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        hud_y += 40
    if current_state == AIRCON:
        cv2.putText(annotated_frame, "AC ON (display only)", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Smart Room Control - Press Q to Quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
transition_to(IDLE)   # Explicitly tears down whatever state we were in
cap.release()
cv2.destroyAllWindows()
if arduino_connected:
    arduino.close()
print("Camera released. Goodbye!")
