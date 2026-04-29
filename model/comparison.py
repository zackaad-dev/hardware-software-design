import os
import time

import numpy as np
import tensorflow as tf

from main import Config, load_and_preprocess_data

TF_MODEL_FILE = "gen/model_reallygood.tflite"
KEKRAS_MODEL_FILE = "gen/model.keras"


def evaluate_models():
    # Load configuration and data
    config = Config()
    _, _, (x_test, y_test), idx_to_class = load_and_preprocess_data(config)

    if x_test.size == 0:
        print("No test data found. Evaluation aborted.")
        return

    print(f"Evaluating models on {len(x_test)} test images...")

    # 1. Keras Baseline
    keras_model = tf.keras.models.load_model(KEKRAS_MODEL_FILE)
    _, keras_accuracy = keras_model.evaluate(x_test, y_test, verbose=0)

    start_time = time.time()
    keras_model.predict(x_test, verbose=0)
    keras_latency = (time.time() - start_time) / len(x_test)
    keras_size = os.path.getsize(KEKRAS_MODEL_FILE)

    # 2. TFLite Evaluation
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]

    # Check if model is quantized (int8)
    is_quantized = input_details["dtype"] == np.int8

    correct_predictions = 0
    start_time = time.time()

    for i in range(len(x_test)):
        input_data = np.expand_dims(x_test[i], axis=0)

        if is_quantized:
            # Rescale input to int8 if necessary
            input_scale, input_zero_point = input_details["quantization"]
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)

        if is_quantized:
            # Output is also quantized
            output_scale, output_zero_point = output_details["quantization"]
            output_data = (
                output_data.astype(np.float32) - output_zero_point
            ) * output_scale

        if np.argmax(output_data) == y_test[i]:
            correct_predictions += 1

    tflite_latency = (time.time() - start_time) / len(x_test)
    tflite_accuracy = correct_predictions / len(x_test)
    tflite_size = os.path.getsize(TF_MODEL_FILE)

    # 3. Output Comparison
    print("\n--- Performance Comparison ---")
    print(
        f"Size: Keras={keras_size / 1024 / 1024:.2f}MB, TFLite={tflite_size / 1024 / 1024:.2f}MB"
    )
    print(
        f"Latency: Keras={keras_latency * 1000:.2f}ms, TFLite={tflite_latency * 1000:.2f}ms"
    )
    print(f"Accuracy: Keras={keras_accuracy:.4f}, TFLite={tflite_accuracy:.4f}")


if __name__ == "__main__":
    evaluate_models()
