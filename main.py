import cv2
import os

detector_rostros = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
reconocedor = cv2.face.LBPHFaceRecognizer_create()


def entrenar_reconocedor(carpeta_fotos):
    rostros = []
    etiquetas = []
    nombres = {}
    id_actual = 0

    for nombre_persona in os.listdir(carpeta_fotos):
        ruta_persona = os.path.join(carpeta_fotos, nombre_persona)
        if not os.path.isdir(ruta_persona):
            continue

        nombres[id_actual] = nombre_persona

        for archivo in os.listdir(ruta_persona):
            ruta_imagen = os.path.join(ruta_persona, archivo)
            imagen = cv2.imread(ruta_imagen)
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            rostros_img = detector_rostros.detectMultiScale(gris, 1.3, 5)

            for (x, y, w, h) in rostros_img:
                rostros.append(gris[y:y + h, x:x + w])
                etiquetas.append(id_actual)

        id_actual += 1

    reconocedor.train(rostros, np.array(etiquetas))
    return nombres

import numpy as np

nombres = entrenar_reconocedor("personas_autorizadas")

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
        rostro = fotograma_gris[y:y + alto, x:x + ancho]

        id_predicho, confianza = reconocedor.predict(rostro)

        if confianza < 70:
            nombre = nombres[id_predicho]
            texto = f"AUTORIZADO: {nombre}"
            color = (0, 255, 0)  # Verde
        else:
            texto = "INTRUSO DETECTADO"
            color = (0, 0, 255)  # Rojo

        cv2.rectangle(fotograma, (x, y), (x + ancho, y + alto), color, 2)
        cv2.putText(fotograma, texto, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Sistema de Reconocimiento", fotograma)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()