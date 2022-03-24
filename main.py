import cv2
from time import time
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
face_mesh  = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.3)

def facialLandmarks(frame):
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

    return frame






previousTime = 0
if __name__ == '__main__':

    # Selección de Camara
    cap = cv2.VideoCapture(1)

    while True:
        # Capturamos Feed
        ret, frame = cap.read()

        # Alto y Ancho de la frame
        width = int(cap.get(3))
        height = int(cap.get(4))

        # TODO Analisis de la Imagen

        frame = facialLandmarks(frame)

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
