import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows backend
if not cap.isOpened():
    raise RuntimeError("Could not open webcam on Windows")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to capture frame")

cv2.imwrite("webcam_capture.jpg", frame)
print("Saved webcam_capture.jpg")