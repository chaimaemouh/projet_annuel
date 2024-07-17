import os
import tensorflow as tf
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseBadRequest
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.layers import BatchNormalization

# Assurez-vous d'importer BatchNormalization si vous utilisez un modèle qui en contient
from tensorflow.keras.layers import BatchNormalization

# Charger le modèle
MODEL_PATH = os.path.join(settings.BASE_DIR, 'mobilenetv2_model.h5')
custom_objects = {'BatchNormalization': BatchNormalization}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

# Dictionnaire des classes
class_dict = {0: 'Glass', 1: 'Plastic', 2: 'Aluminium', 3: 'Organic', 4: 'Carton', 5: 'E-waste'}

def upload_photo(request):
    if request.method == 'POST':
        photo = request.FILES.get('photo')
        if not photo:
            return HttpResponseBadRequest("Aucun fichier téléchargé.")

        fs = FileSystemStorage()
        filename = fs.save(photo.name, photo)
        file_url = fs.url(filename)

        # Préparation de l'image pour la prédiction
        img_path = os.path.join(settings.MEDIA_ROOT, filename)
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisation

        # Prédiction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_dict[predicted_class]

        return render(request, 'myapp/upload_success.html', {
            'file_url': file_url,
            'predicted_label': predicted_label
        })
    return render(request, 'myapp/upload_photo.html')
