import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd

# Definir rutas
train_dir = 'data2/TRAIN'
test_dir = 'data2/TEST'

# Preprocesamiento de imágenes
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

# Construcción del modelo
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 clases: plástico, papel, aluminio
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=80,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=20)

# Guardar el modelo entrenado
model.save('modelo_clasificador_residuos.h5')

# Visualización de la precisión y la pérdida

# Convertir el historial a un DataFrame
history_df = pd.DataFrame(history.history)

# Graficar la precisión
plt.figure(figsize=(8, 6))
plt.plot(history_df['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history_df['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# Graficar la pérdida
plt.figure(figsize=(8, 6))
plt.plot(history_df['loss'], label='Pérdida de Entrenamiento')
plt.plot(history_df['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
