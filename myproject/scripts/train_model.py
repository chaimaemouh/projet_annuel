import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

# Pour éviter les erreurs "OSError: broken data stream when reading image file"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fonction pour charger les données
def load_data_from_directory(dirs, image_size=(128, 128)):
    data = []
    labels = []

    label_mapping = {
        'Glass': 0,
        'Plastic': 1,
        'Aluminium': 2,
        'Organic': 3,
        'Carton': 4,
        'e-waste': 5,
    }

    for dir_path in dirs:
        dir_name = os.path.basename(dir_path)
        label = label_mapping.get(dir_name, -1)

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                try:
                    img = Image.open(file_path)
                    if img.mode in ('P', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        img = img.convert('RGBA')
                    img = img.convert('RGB')
                    img_resized = img.resize(image_size)
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_normalized = img_array / 255.0
                    data.append(img_normalized)
                    labels.append(label)
                except (OSError, ValueError) as e:
                    print(f"Skipping corrupted image: {file_path} - {e}")

    return np.array(data, dtype=np.float32), np.array(labels)

# Chemins des répertoires

dirs = [
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/Glass',
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/Plastic',
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/Carton',
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/Organic',
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/Aluminium',
    'C:/Users/chaimae/Projet_Annuel_4AIBD-master/DataSet_2024/e-waste',
]

# Charger les données avec une taille d'image de 128x128
data, labels = load_data_from_directory(dirs, image_size=(128, 128))

# Convertir les labels en one-hot encoding
labels = to_categorical(labels, num_classes=6)

# Séparer les données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Créer un générateur de données avec augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Préparer le générateur de données pour l'entraînement
datagen.fit(train_data)

# Charger le modèle MobileNetV2 pré-entraîné sans les couches de classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Geler les couches de MobileNetV2
for layer in base_model.layers:
    layer.trainable = False

# Ajouter les couches de classification personnalisées
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                    epochs=20,
                    validation_data=(test_data, test_labels))

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

# Sauvegarder le modèle
model.save('C:/Users/chaimae/Projet_Annuel_4AIBD-master/myproject/mobilenetv2_model.h5')
print("Modèle sauvegardé avec succès.")

# Afficher les courbes de performance
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.show()

# Afficher les courbes de performance
plot_history(history, 'MobileNetV2 Transfer Learning Model')
