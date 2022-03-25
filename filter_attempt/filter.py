import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
nose_image = cv2.imread("sonrisa_luismi.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)
# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        # Nose coordinates
        top_nose = (landmarks.part(52).x, landmarks.part(52).y)
        bot_nose = (landmarks.part(57).x, landmarks.part(57).y)
        left_nose = (landmarks.part(48).x, landmarks.part(48).y)
        right_nose = (landmarks.part(54).x, landmarks.part(54).y)
        center_nose = (landmarks.part(62).x, landmarks.part(62).y)
        nose_width = round(abs(left_nose[0] - right_nose[0] * 1.02))
        nose_height = round(abs(top_nose[1] - bot_nose[1] * 1.02))
        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                    int((center_nose[1] - nose_height / 2) + (0.1 * nose_height)))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                        int((center_nose[1] + nose_height / 2) + (0.1 * nose_height)))
        # Adding the new nose
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 254, 255, cv2.THRESH_BINARY)
        _, nose_mask_inv = cv2.threshold(nose_pig_gray, 254, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                          top_left[0]: top_left[0] + nose_width]
        cv2.imshow("Nose mask", nose_mask)

        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask = nose_mask)
        nose_img = cv2.bitwise_and(nose_pig, nose_pig, mask=nose_mask_inv)
        cv2.imshow("Nose mask", nose_area_no_nose)
        final_nose = cv2.add(nose_img, nose_area_no_nose)
        cv2.imshow("Nose area", nose_area)

        frame[top_left[1]: top_left[1] + nose_height,
              top_left[0]: top_left[0] + nose_width] = final_nose
        cv2.imshow("Nose pig", nose_pig)
        cv2.imshow("final nose", final_nose)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
