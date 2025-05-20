import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# Cargar modelo
modelo = tf.keras.models.load_model('modelo_clasificador_residuos.h5')
clases = ['papel', 'plastico', 'aluminio']

# Conexión con Arduino
arduino = serial.Serial('COM3', 9600) 
time.sleep(2)

# Cámara
cap = cv2.VideoCapture(1)

# Tiempo de espera entre clasificaciones
tiempo_ultimo = time.time() - 60  # Permite clasificar al iniciar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (256, 256))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = modelo.predict(img)[0]
    idx = np.argmax(pred)
    confianza = pred[idx]

    etiqueta = f"{clases[idx]}: {confianza:.2f}"
    cv2.putText(frame, etiqueta, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow("Clasificador de residuos", frame)

    # Si pasa 1 minuto y confianza > 60%
    if confianza >= 0.6 and (time.time() - tiempo_ultimo) > 15:
        arduino.write(clases[idx].encode())  
        tiempo_ultimo = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
