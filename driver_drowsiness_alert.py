import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import random
# Import Chinese Warning voices
voices = ['com.apple.speech.synthesis.voice.mei-jia',
'com.apple.speech.synthesis.voice.sin-ji.premium',
'com.apple.speech.synthesis.voice.ting-ting'];

# Initialization of pyttsx3 for audio alert message to be delivered
engine = pyttsx3.init()
engine.setProperty('voice', random.choice(voices))

# Setting up camera to 0. It should NOT be 1, otherwise error is reported.
cap = cv2.VideoCapture(0)

# Initialization of face detector
face_detector = dlib.get_frontal_face_detector()

# Initialization of face_landmarks for prediction of landmarks of a face
dlib_facelandmark = dlib.shape_predictor("/Users/weicongsu/PycharmProjects/Drivers_Alert/shape_predictor_68_face_landmarks.dat")

# Set up function for calculating the aspect ratio
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

# MAIN LOOP WILL RUN UNTIL THE PROGRAM IS BEING KILLED BY THE USER
drowsiness_level = 0 # counter to store drowsiness level
while True:
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale) # returns empty list if no face is detected
    if len(faces) == 0:
        # if no face is detected, it is usually the case that the head is down, drowsiness level +1
        drowsiness_level += 1
    else:
        # faces are detected, we calculate for each face (usually only 1 face, i.e. the driver's face) the eye closing level
        for face in faces:
            face_landmarks = dlib_facelandmark(gray_scale, face)
            leftEye = []
            rightEye = []

            # Left eye from 42 to 47
            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                next_point = n + 1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 255), 1)

            # Right eye from 36 to 41
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n + 1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

            # Calculate the average aspect ratio
            right_Eye = Detect_Eye(rightEye)
            left_Eye = Detect_Eye(leftEye)
            Eye_Rat = (left_Eye + right_Eye) / 2
            Eye_Rat = round(Eye_Rat, 2)

            # Threshold of 0.2 to decide if drowsiness is detected
            if Eye_Rat < 0.2:
                # eyes closing, drowsiness level +1
                drowsiness_level += 1
            else:
                # awaken, drowsiness level reset to 0
                drowsiness_level = 0

    if drowsiness_level > 5: # 5 is the threshold to avoid false alarm when blinking, it can be reset to other numbers
        cv2.putText(frame, "Drowsiness Detected!", (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 50, 210), 3)
        cv2.putText(frame, "Alert!!!! WAKE UP!!!", (50, 450),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

        # Raise sound alert. Here Chinese version is used.
        engine.say("疲劳驾驶请注意！")
        engine.runAndWait()
    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
    key = cv2.waitKey(5) # wait 5 milliseconds
    if key == ord('q'): # press 'q' to terminate camera
       break
cap.release()
cv2.destroyAllWindows()