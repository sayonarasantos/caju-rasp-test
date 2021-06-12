import tensorflow as tf
import numpy as np


INPUT = [0, 0, 0.47107438, 0.40298507, 0.5, 0.83333333,
             0.29490864, 0.54847278, 1, 0.0873965]
MODEL_PATH = 'model_k_1_m_0.h5'


model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

results = model.evaluate(np.array([INPUT]))
print(results)

predictions = model.predict(np.array([INPUT]))
print(predictions.shape)
