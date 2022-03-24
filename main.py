import cv2
from time import time


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






        '''Constantes a Dibujar al Final'''
        # Contador de FPS
        currentTime = time()
        frames_per_second = 1.0 / (currentTime - previousTime)
        cv2.putText(frame, f'FPS: { int (frames_per_second)}', (10, 30),
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
