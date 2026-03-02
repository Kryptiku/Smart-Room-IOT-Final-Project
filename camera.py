from ultralytics import YOLO
import cv2
import time

# Load pre-trained YOLOv8 model (nano version for speed)
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Will auto-download if not present

# Capture from laptop camera (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

# ── Time-based occupancy validation ──────────────────────────────────────────
# Seconds a person must be continuously detected before counting as an occupant.
OCCUPANCY_THRESHOLD = 5.0  # adjust between 5–10 s as needed

# {track_id: timestamp when this ID was first seen in the current streak}
person_first_seen: dict[int, float] = {}

# Track IDs that have passed the threshold and are confirmed occupants
confirmed_persons: set[int] = set()
# ─────────────────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    current_time = time.time()

    # Use YOLO tracker (ByteTrack) so each person keeps a persistent ID
    results = model.track(frame, persist=True, verbose=False, classes=[0])

    # ── Collect IDs visible in this frame ────────────────────────────────────
    current_ids: set[int] = set()
    if results[0].boxes.id is not None:
        for track_id in results[0].boxes.id.int().tolist():
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

    confirmed_count = len(confirmed_persons)
    total_detected  = len(current_ids)

    # ── Draw annotated frame with bounding boxes ─────────────────────────────
    annotated_frame = results[0].plot()

    # Draw a per-person timer progress bar just below each bounding box
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id.int().tolist()):
            elapsed   = current_time - person_first_seen.get(track_id, current_time)
            progress  = min(elapsed / OCCUPANCY_THRESHOLD, 1.0)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            bar_total  = x2 - x1
            bar_filled = int(bar_total * progress)
            # Orange while waiting, green once confirmed
            bar_color  = (0, 255, 0) if track_id in confirmed_persons else (0, 165, 255)

            # Background track
            cv2.rectangle(annotated_frame, (x1, y2 + 2), (x2, y2 + 14), (50, 50, 50), -1)
            # Filled portion
            cv2.rectangle(annotated_frame, (x1, y2 + 2), (x1 + bar_filled, y2 + 14), bar_color, -1)

            # Status label
            status = "CONFIRMED" if track_id in confirmed_persons else f"{elapsed:.1f}s / {OCCUPANCY_THRESHOLD:.0f}s"
            cv2.putText(annotated_frame, status, (x1, y2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    cv2.putText(annotated_frame,
                f'Detected: {total_detected}   Confirmed: {confirmed_count}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    # Appliance control driven by *confirmed* occupants only
    if confirmed_count > 1:
        cv2.putText(annotated_frame, 'Aircon ON',
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if confirmed_count > 3:
            cv2.putText(annotated_frame, 'Light ON',
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('People Detection - Press Q to Quit', annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")
