import cv2
import numpy as np
import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)
left_eye = cv2.imread("ojo_izq.png")
right_eye = cv2.imread("ojo_der.png")
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
        top_left_eye = (landmarks.part(43).x, landmarks.part(43).y)
        bot_left_eye = (landmarks.part(47).x, landmarks.part(47).y)
        left_left_eye = (landmarks.part(42).x, landmarks.part(42).y)
        right_left_eye = (landmarks.part(45).x, landmarks.part(45).y)

        left_eye_width = round(abs(left_left_eye[0] - right_left_eye[0] * 1.02))
        left_eye_height = round(abs(top_left_eye[1] - bot_left_eye[1] * 1.02))

        top_right_eye = (landmarks.part(38).x, landmarks.part(38).y)
        bot_right_eye = (landmarks.part(40).x, landmarks.part(40).y)
        left_right_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_right_eye = (landmarks.part(39).x, landmarks.part(39).y)

        right_eye_width = round(abs(left_right_eye[0] - right_right_eye[0] * 1.02))
        right_eye_height = round(abs(top_right_eye[1] - bot_right_eye[1] * 1.02))
        # New left nose position
        left_e_top_left = (left_left_eye[0]), int((right_left_eye[1] + left_eye_height / 2))
        left_e_bottom_right = (right_left_eye[0]), int((right_left_eye[1] - left_eye_height / 2))
        # New right nose position
        right_e_top_left = int((left_right_eye[0])), int((right_right_eye[1] + right_eye_height / 2))
        right_e_bottom_right = int((right_right_eye[0])), int((right_right_eye[1] - right_eye_height / 2))

        # Adding the new left eye

        l_eye = cv2.resize(left_eye, (left_eye_width, left_eye_height))
        l_eye_gray = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        _, l_eye_mask = cv2.threshold(l_eye_gray, 254, 255, cv2.THRESH_BINARY)
        _, l_eye_mask_inv = cv2.threshold(l_eye_gray, 254, 255, cv2.THRESH_BINARY_INV)

        l_eye_area = frame[left_e_top_left[1]: left_e_top_left[1] - left_eye_height,
                           left_e_top_left[0]: left_e_top_left[0] + left_eye_width]
        cv2.imshow("l_eye_mask", l_eye_mask)

        l_eye_area_no_eye = cv2.bitwise_and(l_eye_area, l_eye_area, mask = l_eye_mask)
        l_eye_no_back = cv2.bitwise_and(l_eye, l_eye, mask=l_eye_mask_inv)
        cv2.imshow("l_Eye no eye", l_eye_area_no_eye)
        final_l_eye = cv2.add(l_eye_no_back, l_eye_area_no_eye)
        #cv2.imshow("Nose area", l_eye_area)

        #frame[left_e_top_left[1]: left_e_top_left[1] + left_eye_height,
        #      left_e_top_left[0]: left_e_top_left[0] + left_eye_width] = final_nose
        cv2.imshow("Nose pig", l_eye)
        #cv2.imshow("final nose", final_l_eye)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break