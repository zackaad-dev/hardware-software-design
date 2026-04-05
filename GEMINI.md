# Hardware-Software Co-Design for Face Recognition

This project implements a complete pipeline for a face recognition system, from model training and tuning in Python to real-time inference on an ESP32-S3 microcontroller.

## Project Structure

- **`model/`**: Python-based machine learning environment.
  - `main.py`: Training and evaluation script using MobileNetV2 and TensorFlow/Keras.
  - `tune.py`: Hyperparameter tuning using `keras-tuner`.
  - `utils/export_tflite.py`: Tooling to export the Keras model to a quantized TFLite format for microcontrollers.
  - `pyproject.toml` & `uv.lock`: Dependency management using `uv`.
- **`esp32/`**: ESP-IDF project for deployment.
  - `main/main.cpp`: Main application loop, camera handling, and LED control.
  - `main/inference.cpp`: TFLite Micro wrapper for running model inference.
  - `main/model.c`: The exported TFLite model as a byte array.
  - `managed_components/`: ESP-IDF components including `esp-tflite-micro`, `esp-nn`, and `esp32-camera`.

## Technologies

- **Machine Learning**: TensorFlow, Keras, MobileNetV2, Keras Tuner.
- **Embedded**: ESP-IDF (v5.x+), TensorFlow Lite Micro (TFLM), ESP-NN (hardware acceleration), ESP32-S3.
- **Tooling**: `uv` (Python), `idf.py` (ESP-IDF).

## Getting Started

### 1. Model Development (Python)

Navigate to the `model/` directory.

- **Setup**:
  ```bash
  uv sync
  ```
- **Train and Evaluate**:
  ```bash
  uv run python main.py
  ```
- **Hyperparameter Tuning**:
  ```bash
  uv run python tune.py
  ```

The model expects data organized in folders by class name within a `DATA_DIR`.

### 2. Firmware Development (ESP32)

Navigate to the `esp32/` directory. Ensure you have the ESP-IDF environment sourced.

- **Build**:
  ```bash
  idf.py build
  ```
- **Flash and Monitor**:
  ```bash
  idf.py flash monitor
  ```

### 3. Model Export

To update the model on the ESP32:
1. Train the model in the `model/` directory.
2. Export the model to TFLite (quantized to int8).
3. Convert the `.tflite` file to a C array (e.g., using `xxd -i`) and update `esp32/main/model.c` and `esp32/main/model.h`.
4. Ensure `NUM_CLASSES` and `IMG_SIZE` in the firmware match the exported model.

## Development Conventions

- **Python**: Follows standard Keras/TensorFlow patterns. Dependencies are strictly managed by `uv`.
- **C++**: Follows ESP-IDF coding style. Uses `ESP_LOG` for debugging.
- **Memory Management**: The TFLite tensor arena is allocated in **PSRAM** (`MALLOC_CAP_SPIRAM`) to accommodate the MobileNetV2 model.
- **Performance**: Uses `esp-nn` for optimized kernels on the ESP32-S3.

## Hardware Requirements

- **ESP32-S3** with at least 8MB PSRAM (e.g., XIAO ESP32-S3 Sense, Freenove ESP32-S3).
- **Camera Module** (OV2640 supported by default configuration).
- **LED** on GPIO 21 (configurable in `main.cpp`).
