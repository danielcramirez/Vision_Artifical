import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

# Configuración del modelo
img_width, img_height = 150, 150
batch_size = 32
epochs = 10

# Inicialización de rutas para maduros y no maduros
maduros_dir = None
no_maduros_dir = None
dataset_dir = "C:/Users/USER/Documents/Python/Vision_artifical/dataset/"

# Función para cargar imágenes de tomates maduros
def cargar_maduros():
    ruta_maduros = filedialog.askdirectory()
    if ruta_maduros:
        messagebox.showinfo("Cargar", f"Imágenes de tomates maduros cargadas desde: {ruta_maduros}")
        global maduros_dir
        maduros_dir = ruta_maduros

# Función para cargar imágenes de tomates no maduros (biches)
def cargar_no_maduros():
    ruta_no_maduros = filedialog.askdirectory()
    if ruta_no_maduros:
        messagebox.showinfo("Cargar", f"Imágenes de tomates no maduros cargadas desde: {ruta_no_maduros}")
        global no_maduros_dir
        no_maduros_dir = ruta_no_maduros

# Función para entrenar el modelo con las imágenes cargadas
def entrenar_modelo():
    if not maduros_dir or not no_maduros_dir:
        messagebox.showwarning("Error", "Por favor, carga imágenes de tomates maduros y no maduros antes de entrenar.")
        return
    
    # Preparación de los datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,  # Ruta raíz del dataset
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'  # Cambiamos a 'categorical' para varias clases
    )

    # Definición del modelo
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(2, activation='softmax')  # Cambiamos la salida a 2 clases (rojo y verde)
    ])

    # Compilación del modelo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(train_generator, epochs=epochs, verbose=1)

    # Guardar el modelo entrenado
    model.save('tomate_maduro_model.h5')

    messagebox.showinfo("Entrenamiento", "Entrenamiento completo y modelo guardado.")

# Función para mostrar la cámara en la interfaz gráfica
def usar_camara():
    try:
        model = tf.keras.models.load_model('tomate_maduro_model.h5')
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
        return

    cap = cv2.VideoCapture(0)

    def mostrar_frame():
        ret, frame = cap.read()
        if ret:
            # Preprocesar la imagen
            resized_frame = cv2.resize(frame, (img_width, img_height))
            normalized_frame = resized_frame / 255.0
            input_data = normalized_frame.reshape(1, img_width, img_height, 3)

            # Hacer la predicción
            prediction = model.predict(input_data)
            label = 'Tomate Maduro' if prediction[0][0] > 0.5 else 'Tomate No Maduro'

            # Mostrar el resultado en el frame
            color = (0, 255, 0) if label == 'Tomate Maduro' else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Convertir la imagen para mostrar en Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
        lbl_video.after(10, mostrar_frame)

    # Mostrar el video en el área definida en la interfaz
    lbl_video.after(10, mostrar_frame)

# Interfaz gráfica con Tkinter
app = tk.Tk()
app.title("Detección de Tomates")
app.geometry("800x600")

# Botones para cargar imágenes y entrenar el modelo
btn_cargar_maduros = tk.Button(app, text="Cargar Tomates Maduros", command=cargar_maduros)
btn_cargar_maduros.pack(pady=10)

btn_cargar_no_maduros = tk.Button(app, text="Cargar Tomates No Maduros", command=cargar_no_maduros)
btn_cargar_no_maduros.pack(pady=10)

btn_entrenar = tk.Button(app, text="Entrenar Modelo", command=entrenar_modelo)
btn_entrenar.pack(pady=10)

# Área para mostrar la cámara
lbl_video = tk.Label(app)
lbl_video.pack(pady=20)

# Botón para activar la cámara y hacer la detección
btn_camara = tk.Button(app, text="Usar Cámara para Detección", command=usar_camara)
btn_camara.pack(pady=10)

# Ejecutar la aplicación
app.mainloop()
