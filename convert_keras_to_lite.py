import tensorflow as tf
import numpy as np
import os


GROUP = 'caju'
KERAS_MODELS_DIR = f'./K_{GROUP}/model_k_1_m_'


os.system(f'mkdir L_{GROUP}')

for model_number in range(27):
    keras_model_path = f'{KERAS_MODELS_DIR}{model_number}.h5'
    keras_model = tf.keras.models.load_model(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    lite_model = converter.convert()

    with open(f'L_{GROUP}/model_{model_number}.tflite', 'wb') as f:
        f.write(lite_model)
