import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys
import os
import logging

# Configuración del registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ruta del modelo
model_path = '/Users/nfanlo/dev/technical-test/part1/models/fine_tuned_inceptionv3.h5'

# Función para cargar el modelo
def load_best_model(model_path):
    """
    Function that loads the best trained model in the project notebook.

    The function waits for the model path defined previously and returns 
    the model loaded in the system.
    """
    try:
        model = load_model(model_path)
        logger.info(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        sys.exit(1)

model = load_best_model(model_path)

def preprocess_image(image_path):
    """
    Function that preprocesses the image so that it has the appropriate 
    size and shape for the model (150x150).

    The function waits for the input of the path of the image to be preprocessed 
    and returns the preprocessed image.
    """
    try:
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Ajusta esto según tu modelo
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        logger.info(f'Image {image_path} processed successfully')
        return image
    except Exception as e:
        logger.error(f'Error processing image {image_path}: {e}')
        sys.exit(1)

def predict(image_path):
    """
    Function that makes a prediction on the preprocessed image.

    The function waits for the path of the preprocessed image and returns a str 
    with the model prediction between Cat/Dog.
    """
    try:
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        logger.info(f'Prediction for image {image_path} made successfully')
        return prediction
    except Exception as e:
        logger.error(f'Error making prediction for image {image_path}: {e}')
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error('Usage: python predict.py <ruta_a_imagen>')
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        logger.error(f'The file {image_path} does not exist.')
        sys.exit(1)
    
    prediction = predict(image_path)
    print(f"Prediction: {prediction[0][0]}")
    if prediction[0][0] > 0.5:
        print("La imagen es clasificada como: Perro")
    else:
        print("La imagen es clasificada como: Gato")
