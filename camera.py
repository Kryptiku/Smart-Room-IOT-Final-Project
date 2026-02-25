from ultralytics import YOLO
import cv2

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Detect only people (class 0) in the frame
    results = model(frame, verbose=False, classes=[0])
    
    # Count people (class 0 is 'person' in COCO dataset)
    people_count = sum(1 for r in results[0].boxes.cls if int(r) == 0)
    
    # Draw detection boxes and labels on frame
    annotated_frame = results[0].plot()
    
    # Add people count text to the frame
    cv2.putText(annotated_frame, f'People Detected: {people_count}', 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Display the frame
    cv2.imshow('People Detection - Press Q to Quit', annotated_frame)
    # Add people count text to the frame
    cv2.putText(annotated_frame, f'People Detected: {people_count}', 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if people_count > 1:
        cv2.putText(annotated_frame, f'Aircon ON', 
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if people_count > 3:
            cv2.putText(annotated_frame, f'Light ON', 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Display the frame
    cv2.imshow('People Detection - Press Q to Quit', annotated_frame)


    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")
