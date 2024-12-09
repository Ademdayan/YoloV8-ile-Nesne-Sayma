import os
os.environ ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import cv2
import numpy as np

from ultralytics import YOLO
from collections import defaultdict

# Text & Color Params 
thickness = 2

color_black = (0,0,0)
color_red = (0,0,255)
color_green = (0,255,0)
color_white = (255,255,255)

font_scale = 0.7
font = cv2.FONT_HERSHEY_SIMPLEX

# Video Params
video_path = "video/luggage.mp4"
cap = cv2.VideoCapture(video_path)

height, width = 720, 1280
print("[INFO].. Width:", width)
print("[INFO].. Height:", height)

# Model Params
model_name = "models/best.pt"
model = YOLO(model_name)

counter = {}
track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = cv2.resize(frame, (width, height))
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int") # 2.0 -> 2

    cv2.line(frame, (int(width/2), 0), (int(width/2), height), color_red, thickness)
    cv2.rectangle(frame, (int(width/2)+7, int(height/2)+5), (int(width/2)+175, int(height/2)-25), color_white, -1)
    cv2.putText(frame, "Referans Hatti", (int(width/2)+10, int(height/2)), font, font_scale, color_red, thickness)

    # Bagajların ortasına nokta atılacak     
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        text = "ID:{} BAGAJ".format(track_id)
        
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        center_coordinates = (cx, cy)

        if cx > int(width/2) :
            cv2.circle(frame, center_coordinates, 3, color_red, -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color_red, 2)
            cv2.rectangle(frame, (x1,y1), (x1+175,y1-25), color_red, -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_white, 2)
            
        else: 
            cv2.circle(frame, center_coordinates, 3, color_green, -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color_green, 2)
            cv2.rectangle(frame, (x1,y1), (x1+175,y1-25), color_green, -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_white, 2)
            counter[track_id] = x1, y1, x2, y2

        number_of_luggage = len(list(counter.keys()))  

        info = f"TOPLAM GECIS: {number_of_luggage}"
        cv2.rectangle(frame, (5,5), (150,40), color_white, -1)
        cv2.putText(frame, info, (15,30), font, font_scale, color_red, thickness)  

    cv2.imshow("Object Counting", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
