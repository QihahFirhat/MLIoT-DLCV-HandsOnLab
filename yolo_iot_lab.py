from ultralytics import YOLO
import cv2
 
# Load lightweight YOLOv8 model
model = YOLO("yolov8n.pt")
 
# Open webcam
cap = cv2.VideoCapture(0)
 
# Check webcam
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()
 
print("Press 'q' to quit")
 
while True:
 
    # Capture frame
    ret, frame = cap.read()
 
    if not ret:
        print("Failed to capture frame")
        break
 
    # Run object detection
    results = model(frame) #  with confidence threshold: results = model(frame, conf=0.7)

    # Draw detection results
    annotated_frame = results[0].plot()
 
    # Display output
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)
 
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release resources
cap.release()
cv2.destroyAllWindows()
