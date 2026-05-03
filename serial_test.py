import serial
import time
from ultralytics import YOLO
import cv2

# ── Serial Setup ──────────────────────────────────────────────────────────────
try:
    arduino = serial.Serial("COM7", 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    arduino_connected = True
    print("Arduino connected on COM7")
except Exception as e:
    arduino_connected = False
    print(f"Arduino not connected: {e} — running in simulation mode")


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

# ── Debounce Settings ─────────────────────────────────────────────────────────
DEBOUNCE_FRAMES = 8      # Number of consecutive frames the count must be stable
DEBOUNCE_TOLERANCE = 1   # Max allowed difference between frames to be "stable"

# ── OFF Delay Settings ────────────────────────────────────────────────────────
OFF_DELAY_SECONDS = 10.0  # How long to wait after desired state drops before acting

# ── Internal Debounce + Hysteresis Buffers ────────────────────────────────────
count_history = []        # Rolling buffer of raw_count values
stable_count = 0          # Last debounce-confirmed count
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
    Append raw_count to the rolling history buffer.
    Return a stable count only when the last DEBOUNCE_FRAMES values
    are all within DEBOUNCE_TOLERANCE of each other.
    Returns None if the count is not yet stable.
    """
    count_history.append(raw_count)
    if len(count_history) > DEBOUNCE_FRAMES:
        count_history.pop(0)

    if len(count_history) < DEBOUNCE_FRAMES:
        return None  # Not enough history yet

    if max(count_history) - min(count_history) <= DEBOUNCE_TOLERANCE:
        return round(sum(count_history) / len(count_history))

    return None  # Still flickering


# ─────────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()

    results = model(frame, verbose=False, classes=[0], conf=CONFIDENCE_THRESHOLD)

    # ── Room-Level Occupancy Signal ───────────────────────────────────────────
    if results[0].boxes is not None:
        raw_count = len(results[0].boxes)
    else:
        raw_count = 0

    # ── Frame-Based Debounce ──────────────────────────────────────────────────
    new_stable = get_debounced_count(raw_count)
    if new_stable is not None:
        stable_count = new_stable

    target_state = count_to_target_state(stable_count)

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

    # ── Draw Annotated Frame ──────────────────────────────────────────────────
    annotated_frame = results[0].plot()

    # ── HUD Overlay ───────────────────────────────────────────────────────────
    # ── HUD Overlay ───────────────────────────────────────────────────────────
    cv2.putText(
        annotated_frame,
        f"Detected: {raw_count}   Stable: {stable_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
    )

    cv2.putText(
        annotated_frame,
        f"State: {STATE_NAMES[current_state]}   Target: {STATE_NAMES[count_to_target_state(stable_count)]}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 0),
        2,
    )

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
