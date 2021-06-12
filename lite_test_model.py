import numpy as np
import tflite_runtime.interpreter as tf
import random
import os


GROUP = 'caju'
MODEL_PATH = f'./L_{GROUP}/model_0.tflite'
MIN_MAX = [0, 5, 0.5, 0.55, 26.3, 38.6, 21.7, 28.3, 19, 81, 57, 100,
           0.363606, 2.187793, 4.965, 182.5, 99.622083, 99.99875, 0, 108.7]


def create_input(min_max_model):
    scaled_input = list()
    for i in range(0, len(min_max_model), 2):
        random_value = random.uniform(min_max_model[i],
                                      min_max_model[i+1])
        scaled_input.append((random_value - min_max_model[i]) /
                            (min_max_model[i+1] - min_max_model[i]))
    return scaled_input


def test_model(model, data):
    interpreter = None
    interpreter = tf.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, [np.array(data, dtype=np.float32)])
    interpreter.invoke()

    output_index = interpreter.get_output_details()[0]['index']
    output_tensor = interpreter.get_tensor(output_index)

    print(f'Model: {model}\n'
           f'Input: {data}\n'
           f'Output: {output_tensor}\n')


def main():
    random_input = create_input(MIN_MAX)
    test_model(MODEL_PATH, random_input)


if __name__ == '__main__':
    main()
