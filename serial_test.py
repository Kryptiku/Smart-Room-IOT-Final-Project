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

# ── Occupancy Signal Settings ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
CONFIRM_SECONDS = 5.0  # Count must persist this long before it is confirmed.
CONFIRM_DECAY_RATE = 0.7  # Progress decay speed when threshold is not met.

# ── Appliance State + Hysteresis ──────────────────────────────────────────────
appliance_state = {
    "light": False,
    "fan": False,
    "ac": False,
}

ON_THRESHOLDS = {
    "light": 1,
    "fan": 2,
    "ac": 3,
}
OFF_DELAYS = {
    "light": 10.0,
    "fan": 10.0,
    "ac": 10.0,
}

# Confirmation progress is time-accumulated and decays on brief dropouts.
confirm_progress = {
    "light": 0.0,
    "fan": 0.0,
    "ac": 0.0,
}
confirmed_levels = {
    "light": False,
    "fan": False,
    "ac": False,
}
off_started = {
    "light": None,
    "fan": None,
    "ac": None,
}


def set_appliance(name, turn_on):
    """Switch appliance only when its state changes."""
    if appliance_state[name] == turn_on:
        return

    appliance_state[name] = turn_on

    if name == "light":
        send_command("LIGHT_ON" if turn_on else "LIGHT_OFF")
    elif name == "fan":
        send_command("FAN_ON" if turn_on else "FAN_OFF")
    elif name == "ac":
        # No relay for AC yet — display only
        print(f"  [AC] {'ON' if turn_on else 'OFF'} (display only)")


# ─────────────────────────────────────────────────────────────────────────────

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()
    dt = max(0.0, current_time - prev_time)
    prev_time = current_time

    results = model(frame, verbose=False, classes=[0], conf=CONFIDENCE_THRESHOLD)

    # ── Room-Level Occupancy Signal ───────────────────────────────────────────
    if results[0].boxes is not None:
        raw_count = len(results[0].boxes)
    else:
        raw_count = 0

    # ── Confirmation Logic Per Threshold ───────────────────────────────────────
    for appliance_name in ["light", "fan", "ac"]:
        meets_threshold = raw_count >= ON_THRESHOLDS[appliance_name]
        if meets_threshold:
            confirm_progress[appliance_name] = min(
                CONFIRM_SECONDS,
                confirm_progress[appliance_name] + dt,
            )
        else:
            confirm_progress[appliance_name] = max(
                0.0,
                confirm_progress[appliance_name] - (dt * CONFIRM_DECAY_RATE),
            )

        confirmed_levels[appliance_name] = (
            confirm_progress[appliance_name] >= CONFIRM_SECONDS
        )

    confirmed_count = 0
    if confirmed_levels["light"]:
        confirmed_count = 1
    if confirmed_levels["fan"]:
        confirmed_count = 2
    if confirmed_levels["ac"]:
        confirmed_count = 3

    # ── Appliance Decision Engine with OFF Delay Hysteresis ──────────────────
    for appliance_name in ["light", "fan", "ac"]:
        desired_on = confirmed_levels[appliance_name]

        if desired_on:
            off_started[appliance_name] = None
            set_appliance(appliance_name, True)
            continue

        if appliance_state[appliance_name]:
            if off_started[appliance_name] is None:
                off_started[appliance_name] = current_time
            elapsed_off = current_time - off_started[appliance_name]
            if elapsed_off >= OFF_DELAYS[appliance_name]:
                set_appliance(appliance_name, False)
                off_started[appliance_name] = None
        else:
            off_started[appliance_name] = None

    # ── Draw Annotated Frame ──────────────────────────────────────────────────
    annotated_frame = results[0].plot()

    # ── HUD Overlay ───────────────────────────────────────────────────────────
    cv2.putText(
        annotated_frame,
        f"Detected: {raw_count}   Confirmed: {confirmed_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
    )

    cv2.putText(
        annotated_frame,
        f'Progress L/F/A: {confirm_progress["light"]:.1f}/{confirm_progress["fan"]:.1f}/{confirm_progress["ac"]:.1f}s',
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
    )

    hud_y = 110
    if appliance_state["light"]:
        cv2.putText(
            annotated_frame,
            "Light ON",
            (10, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
        )
        hud_y += 40
    if appliance_state["fan"]:
        cv2.putText(
            annotated_frame,
            "Fan ON",
            (10, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            3,
        )
        hud_y += 40
    if appliance_state["ac"]:
        cv2.putText(
            annotated_frame,
            "AC ON (display only)",
            (10, hud_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )

    cv2.imshow("Smart Room Control - Press Q to Quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
send_command("LIGHT_OFF")
send_command("FAN_OFF")
cap.release()
cv2.destroyAllWindows()
if arduino_connected:
    arduino.close()
print("Camera released. Goodbye!")
