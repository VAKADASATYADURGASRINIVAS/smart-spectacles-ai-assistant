
import cv2
import torch
import pyttsx3
import time

# Initialize TTS engine
engine = pyttsx3.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.5  # confidence threshold

# Start video capture
cap = cv2.VideoCapture(0)
prev_alert = ""
last_alert_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]
    labels = detections['name'].tolist()

    alert = ""
    if 'person' in labels:
        alert = "Person ahead, please stop"
    elif 'car' in labels or 'bus' in labels or 'truck' in labels:
        alert = "Vehicle ahead, move left"
    elif not labels:
        alert = "Path is clear"

    # Announce alert every 5 seconds if different
    if alert != prev_alert and time.time() - last_alert_time > 5:
        print("Voice Alert:", alert)
        engine.say(alert)
        engine.runAndWait()
        prev_alert = alert
        last_alert_time = time.time()

    # Show frame
    cv2.imshow("Smart Spectacles View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
