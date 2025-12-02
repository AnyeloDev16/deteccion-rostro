import cv2

detector_rostros = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camara = cv2.VideoCapture(0)

while True:

    lectura_correcta, fotograma = camara.read()

    if not lectura_correcta:
        break

    fotograma_gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    rostros_detectados = detector_rostros.detectMultiScale(
        fotograma_gris,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, ancho, alto) in rostros_detectados:
        cv2.rectangle(
            fotograma,
            (x, y),
            (x + ancho, y + alto),
            (0, 255, 0),
            2
        )

    cv2.imshow("Detección de Rostros", fotograma)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
