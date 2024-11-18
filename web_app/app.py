from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import uuid

app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.load_model('efficientnetb3-Eye Disease-95.77.h5')

# Lista de clases (asegúrate de que el orden es correcto)
classes = ['ARMD', 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Crear el diccionario de índices a clases
index_to_class = {index: label for index, label in enumerate(classes)}

def load_and_preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    # No normalizar la imagen si no lo hiciste durante el entrenamiento
    # img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Comprobar si se ha enviado un archivo
        if 'file' not in request.files:
            return render_template('index.html', message='No se ha seleccionado ningún archivo')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No se ha seleccionado ningún archivo')

        if file:
            # Validar el tipo de archivo
            if not allowed_file(file.filename):
                return render_template('index.html', message='Tipo de archivo no permitido')
            # Generar un nombre de archivo único
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            img_path = os.path.join('static/uploads', filename)
            # Crear la carpeta si no existe
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            file.save(img_path)

            # Preprocesar la imagen
            preprocessed_image = load_and_preprocess_image(img_path, img_size=(224, 224))

            # Realizar la predicción
            prediction = model.predict(preprocessed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = index_to_class[predicted_class_index]

            # Pasar el resultado a la plantilla
            image_url = url_for('static', filename='uploads/' + filename)
            return render_template('result.html', prediction=predicted_class_label, image_path=image_url)

    return render_template('index.html')

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)
