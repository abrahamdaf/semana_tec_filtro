import cv2

if __name__ == '__main__':
    # Selección de Camara
    cap = cv2.VideoCapture(1)

    while True:
        # Capturamos Feed
        ret, frame = cap.read()

        # Mostramos la imagen
        cv2.imshow("Frame", frame)

        # Si presionamos 'q' cerramos el programa
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
