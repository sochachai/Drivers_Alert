import torch
import numpy as np
import cv2
import pyttsx3
engine = pyttsx3.init()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp12/weights/best.pt', force_reload=True)
cap = cv2.VideoCapture(0)

distraction_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)
    status = results.pandas().xyxy[0]['name'].values

    if len(status) > 0:
        if status[0]=='straight':
            distraction_count = 0
        else: distraction_count += 1

    if distraction_count >10: # distraction level is set to 10 to avoid false alarm when driver is watching rearview mirror for a short amount of time
        cv2.putText(frame, "Distraction Driving Detected!", (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
        cv2.putText(frame, "Alert!!!! Pay Attention！", (50, 450),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

        # CALLING THE AUDIO FUNCTION OF TEXT TO AUDIO
        # FOR ALERTING THE PERSON
        engine.say("Warning！Distracted Driving Detected！")
        engine.runAndWait()
    cv2.imshow("Distraction DETECTOR with YOLO", np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

