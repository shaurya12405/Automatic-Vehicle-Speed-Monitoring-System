import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
from IPython.display import Video, display

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  
# Input video path (uploaded earlier)
input_path = "traffic.mp4"
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Calibration
REAL_DISTANCE_METERS = 10
PIXEL_DISTANCE_REF = 300  # Calibrate this for your video

# Tracking dictionary
vehicle_tracker = {}

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if results.boxes.id is None:
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    ids = results.boxes.id.int().cpu().numpy()
    classes = results.boxes.cls.int().cpu().numpy()

    for box, cls, id_ in zip(boxes, classes, ids):
        label = model.names[int(cls)]
        if label not in ['car', 'motorcycle', 'bus', 'truck']:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if id_ in vehicle_tracker:
            prev_cx, prev_cy, prev_time = vehicle_tracker[id_]
            pixel_dist = euclidean((cx, cy), (prev_cx, prev_cy))
            time_elapsed = time.time() - prev_time

            meters = (pixel_dist / PIXEL_DISTANCE_REF) * REAL_DISTANCE_METERS
            speed_kmh = (meters / time_elapsed) * 3.6  # m/s to km/h

            # Draw box and speed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {int(speed_kmh)} km/h", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            vehicle_tracker[id_] = (cx, cy, time.time())
        else:
            vehicle_tracker[id_] = (cx, cy, time.time())

    out.write(frame)

cap.release()
out.release()
