import cv2
from time import time
import mediapipe as mp
import itertools
import numpy as np

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, min_detection_confidence=0.5,
                                            min_tracking_confidence=0.3)
sonrisa_luismi = cv2.imread('./assets/sonrisa_luismi.png')
ojo_der = cv2.imread('./assets/ojo_der.png')
ojo_izq = cv2.imread('./assets/ojo_izq.png')


def getSize(face_landmarks, INDEXES):
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []

    for i in INDEXES_LIST:
        landmarks.append([int(face_landmarks.landmark[i].x * width),
                          int(face_landmarks.landmark[i].y * height)])

    landmarks = np.array(landmarks)

    x, y, width_landmark, height_landmark = cv2.boundingRect(landmarks)

    return width_landmark, height_landmark, landmarks


def overlay(frame, filter, landmarks, INDEXES):
    try:
        filter_h, filter_w, _ = filter.shape
        _, part_height, landmarks = getSize(landmarks, INDEXES)
        required_height = int(part_height * 1.5)
        resized_filter_img = cv2.resize(filter, (int(filter_w * (required_height / filter_h)), required_height))
        filter_h, filter_w, _ = resized_filter_img.shape

        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        center = landmarks.mean(axis=0).astype("int")

        location = (int(center[0] - filter_w / 2), int(center[1] - filter_h / 2))

        ROI = frame[location[1]: location[1] + filter_h, location[0]: location[0] + filter_w]

        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)

        resultant_image = cv2.add(resultant_image, resized_filter_img)

        frame[location[1]: location[1] + filter_h,
        location[0]: location[0] + filter_w] = resultant_image
    except Exception as e:
        pass






def sad_luismi_filter(frame):
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:

        for face, face_landmarks in enumerate(results.multi_face_landmarks):
            overlay(frame=frame, landmarks=face_landmarks, filter=sonrisa_luismi,
                    INDEXES=mp.solutions.face_mesh.FACEMESH_LIPS)
            overlay(frame=frame, landmarks=face_landmarks, filter=ojo_der,
                    INDEXES=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
            overlay(frame=frame, landmarks=face_landmarks, filter=ojo_izq,
                    INDEXES=mp.solutions.face_mesh.FACEMESH_LEFT_EYE)





previousTime = 0
sad_luismi =  True
if __name__ == '__main__':

    # Selección de Camara
    cap = cv2.VideoCapture(1)

    while True:
        # Capturamos Feed
        success, frame = cap.read()


        # Si no logramos conseguir la siguiente frame
        if not success:
            continue
        # Alto y Ancho de la frame
        width = int(cap.get(3))
        height = int(cap.get(4))


        # Activación de filtros
        if cv2.waitKey(1) == ord('w'):
            sad_luismi = not sad_luismi


        if sad_luismi:
            sad_luismi_filter(frame)


        '''Constantes a Dibujar al Final'''
        # Contador de FPS
        currentTime = time()
        frames_per_second = 1.0 / (currentTime - previousTime)
        cv2.putText(frame, f'FPS: {int(frames_per_second)}', (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        previousTime = currentTime
        # Titulo SEMANA TEC
        cv2.putText(frame, 'SEMANA TEC', (width - 250, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 5)

        # Mostramos la imagen
        cv2.imshow("Frame", frame)

        # Si presionamos 'q' cerramos el programa
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()