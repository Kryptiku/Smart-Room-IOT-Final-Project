import serial
import time
from ultralytics import YOLO
import cv2

# ── Serial Setup ──────────────────────────────────────────────────────────────
try:
    arduino = serial.Serial('COM7', 9600, timeout=1)
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
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

# ── Occupancy Validation Settings ────────────────────────────────────────────
OCCUPANCY_THRESHOLD = 5.0  # seconds before confirming a person

person_first_seen: dict[int, float] = {}
confirmed_persons: set[int] = set()

# ── Appliance State + Cooldown ────────────────────────────────────────────────
appliance_state = {
    "light": False,
    "fan":   False,
    "ac":    False,
}
COOLDOWN = 3.0  # seconds to wait before switching appliance state again
last_switch_time = {
    "light": 0.0,
    "fan":   0.0,
    "ac":    0.0,
}

def set_appliance(name, turn_on, current_time):
    """Only switch if state changed and cooldown has passed."""
    if appliance_state[name] == turn_on:
        return  # Already in desired state, do nothing
    if current_time - last_switch_time[name] < COOLDOWN:
        return  # Cooldown not finished yet

    appliance_state[name] = turn_on
    last_switch_time[name] = current_time

    if name == "light":
        send_command("LIGHT_ON" if turn_on else "LIGHT_OFF")
    elif name == "fan":
        send_command("FAN_ON" if turn_on else "FAN_OFF")
    elif name == "ac":
        # No relay for AC yet — display only
        print(f"  [AC] {'ON' if turn_on else 'OFF'} (display only)")

# ─────────────────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()

    results = model.track(frame, persist=True, verbose=False, classes=[0])

    # ── Collect visible IDs ───────────────────────────────────────────────────
    current_ids: set[int] = set()
    if results[0].boxes.id is not None:
        for track_id in results[0].boxes.id.int().tolist():
            current_ids.add(track_id)
            if track_id not in person_first_seen:
                person_first_seen[track_id] = current_time
                print(f"  [tracker] Person ID {track_id} appeared – timer started.")

    # ── Reset timers for lost IDs ─────────────────────────────────────────────
    lost_ids = set(person_first_seen.keys()) - current_ids
    for lost_id in lost_ids:
        elapsed = current_time - person_first_seen[lost_id]
        print(f"  [tracker] Person ID {lost_id} lost after {elapsed:.1f}s – timer reset.")
        del person_first_seen[lost_id]
        confirmed_persons.discard(lost_id)

    # ── Promote long-enough IDs to confirmed ──────────────────────────────────
    for track_id in current_ids:
        elapsed = current_time - person_first_seen[track_id]
        if elapsed >= OCCUPANCY_THRESHOLD and track_id not in confirmed_persons:
            confirmed_persons.add(track_id)
            print(f"  [tracker] Person ID {track_id} CONFIRMED ({elapsed:.1f}s).")

    confirmed_count = len(confirmed_persons)
    total_detected  = len(current_ids)

    # ── Appliance Decision Engine ─────────────────────────────────────────────
    # Level 1: 1+ confirmed → Light ON
    set_appliance("light", confirmed_count >= 1, current_time)

    # Level 2: 2+ confirmed → Fan ON
    set_appliance("fan", confirmed_count >= 2, current_time)

    # Level 3: 3+ confirmed → AC ON (display only for now)
    set_appliance("ac", confirmed_count >= 3, current_time)

    # Turn everything off if room is empty
    if confirmed_count == 0:
        set_appliance("light", False, current_time)
        set_appliance("fan",   False, current_time)
        set_appliance("ac",    False, current_time)

    # ── Draw Annotated Frame ──────────────────────────────────────────────────
    annotated_frame = results[0].plot()

    # Per-person timer progress bar
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id.int().tolist()):
            elapsed  = current_time - person_first_seen.get(track_id, current_time)
            progress = min(elapsed / OCCUPANCY_THRESHOLD, 1.0)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            bar_total  = x2 - x1
            bar_filled = int(bar_total * progress)
            bar_color  = (0, 255, 0) if track_id in confirmed_persons else (0, 165, 255)

            cv2.rectangle(annotated_frame, (x1, y2 + 2),  (x2, y2 + 14), (50, 50, 50), -1)
            cv2.rectangle(annotated_frame, (x1, y2 + 2),  (x1 + bar_filled, y2 + 14), bar_color, -1)

            status = "CONFIRMED" if track_id in confirmed_persons else f"{elapsed:.1f}s / {OCCUPANCY_THRESHOLD:.0f}s"
            cv2.putText(annotated_frame, status, (x1, y2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    # ── HUD Overlay ───────────────────────────────────────────────────────────
    cv2.putText(annotated_frame,
                f'Detected: {total_detected}   Confirmed: {confirmed_count}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    hud_y = 80
    if appliance_state["light"]:
        cv2.putText(annotated_frame, 'Light ON',
                    (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        hud_y += 40
    if appliance_state["fan"]:
        cv2.putText(annotated_frame, 'Fan ON',
                    (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        hud_y += 40
    if appliance_state["ac"]:
        cv2.putText(annotated_frame, 'AC ON (display only)',
                    (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow('Smart Room Control - Press Q to Quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
send_command("LIGHT_OFF")
send_command("FAN_OFF")
cap.release()
cv2.destroyAllWindows()
if arduino_connected:
    arduino.close()
print("Camera released. Goodbye!")