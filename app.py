import cv2
import torch
import numpy as np
import pytesseract
from flask import Flask, render_template, Response, request
from gtts import gTTS
import os
import uuid
import threading
import time
import pygame

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
pygame.mixer.init()

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.conf = 0.4

last_spoken_time = 0
SPEAK_INTERVAL = 5  # seconds

def speak(text):
    def thread_speak():
        try:
            tts = gTTS(text=text, lang='en')
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove(filename)
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=thread_speak).start()

def detect_text(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    return text

def gen_frames(mode):
    global last_spoken_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        h, w = frame.shape[:2]

        do_nav = mode in ["navigation", "both"]
        do_text = mode in ["text", "both"]

        left_area = 0
        right_area = 0
        front_obstacle = False

        if do_nav:
            results = model(frame)
            labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

            for i, label in enumerate(labels):
                row = cords[i]
                x1, y1, x2, y2, conf = row[:5]
                if conf < 0.4:
                    continue

                left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                object_width = right - left
                object_height = bottom - top
                object_area = object_width * object_height

                center_x = (left + right) / 2
                if center_x < w / 2:
                    left_area += object_area
                else:
                    right_area += object_area

                if object_height / h > 0.6 and 0.3 < center_x / w < 0.7:
                    front_obstacle = True

                class_name = model.names[int(label)]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            total_area = w * h
            left_ratio = left_area / (total_area / 2)
            right_ratio = right_area / (total_area / 2)

            if current_time - last_spoken_time >= SPEAK_INTERVAL:
                if front_obstacle:
                    speak("Obstacle ahead. Please stop.")
                elif left_ratio > 0.2:
                    speak("Move right")
                elif right_ratio > 0.2:
                    speak("Move left")
                else:
                    speak("Go forward")
                last_spoken_time = current_time

        if do_text:
            roi = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            text = detect_text(roi)
            if text and current_time - last_spoken_time >= SPEAK_INTERVAL:
                speak(text)
                last_spoken_time = current_time
            cv2.rectangle(frame, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    mode = request.args.get("mode", default="navigation")
    return Response(gen_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)
