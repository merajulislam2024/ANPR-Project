from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import re
import os
from paddleocr import PaddleOCR
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("model_weight/license_plate_detector.pt")

className = ["License_Plate"]

cap = cv.VideoCapture("data/carLicence4.mp4")

OCR = PaddleOCR(use_angle_cls = True, use_gpu = False)

def ocr(frame, x1, y1, x2, y2):

    frame = frame[y1:y2, x1: x2]

    result = OCR.ocr(frame, det=False, rec = True, cls = False)
    text = ""

    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 90:
            text = r[0][0]

    text = text.replace("O", "0").replace("I", "1").replace("粤", "")

    # Remove non-alphanumeric characters
    text = re.sub(r'[\W_]', '', text)  # Removes non-alphanumeric and underscores

    # Validate if the text is purely alphanumeric and has a valid length
    if re.match(r'^[A-Z0-9]+$', text) and 6 <= len(text) <= 8:
        return text  # Return cleaned text if valid

    return ""  # Return an empty string for invalid plates



while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h

            conf = math.ceil(box.conf[0]*100)/100

            cls = int(box.cls[0])
            currentClass = className[cls]

            # crop
            roi = frame[y1:y1+h, x1:x1+w]

            text = ocr(frame, x1, y1, x2, y2)

            if currentClass == "License_Plate" and conf > 0.5:
                cvzone.cornerRect(frame, bbox, l=1, t=1, rt=2, colorR=(255, 0, 0), colorC=(255, 0, 0))
                cvzone.putTextRect(frame, f"{text}", (max(0, bbox[0]), max(35, bbox[1])), 1, 1, colorR=(255, 0, 0), offset=1)


    cv.imshow("ANPR System", frame)
    # cv.imshow("ANPR System", roi_rgb)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()