import argparse
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@dataclass
class Config:
    # Directories
    DATA_DIR: str = os.path.expanduser("~/personal/data")

    # Image Settings
    IMG_SIZE: int = 224

    # Hyperparameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 64
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.2
    MOBILE_NET_ALPHA: float = 0.5
    DENSE_UNITS: int = 128

    PATIENCE: int = 15


def load_image_as_array(filepath, img_size):
    """Loads an image and resizes it, returning a numpy array."""
    try:
        img = Image.open(filepath).convert("RGB")
        img = img.resize((img_size, img_size))
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def get_data_from_folder(folder_path, label, img_size):
    """Loads all valid images from a folder with a given label."""
    x, y = [], []
    valid_extensions = (".jpg", ".jpeg", ".png", ".heic", ".webp")
    if not os.path.exists(folder_path):
        return x, y

    for fp in glob.glob(os.path.join(folder_path, "*")):
        if fp.lower().endswith(valid_extensions):
            img_arr = load_image_as_array(fp, img_size)
            if img_arr is not None:
                x.append(img_arr)
                y.append(label)
    return x, y


def load_and_preprocess_data(config: Config):
    """
    Dynamically loads training, validation, and test data based on folder structure.
    Expects DATA_DIR to contain 'training', 'validaton', and 'test' subfolders.
    Each subfolder should contain class-named folders.
    """
    base_dir = config.DATA_DIR
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Data directory '{base_dir}' not found.")

    train_dir = os.path.join(base_dir, "training")
    val_dir = os.path.join(base_dir, "validation")
    test_dir = os.path.join(base_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' not found.")

    # Identify classes from the training folder
    classes = sorted(
        [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    )
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    if not classes:
        raise ValueError(f"No class folders found in {train_dir}")

    print(f"Detected Classes: {classes}")

    def load_set(subset_dir):
        x_set, y_set = [], []
        for class_name in classes:
            label = class_to_idx[class_name]
            folder = os.path.join(subset_dir, class_name)
            x, y = get_data_from_folder(folder, label, config.IMG_SIZE)
            x_set.extend(x)
            y_set.extend(y)
        return x_set, y_set

    # Load sets
    train_x, train_y = load_set(train_dir)
    val_x, val_y = load_set(val_dir)
    test_x, test_y = load_set(test_dir)

    if not train_x:
        raise ValueError(f"No training images found in {train_dir}")

    # Helper to preprocess and convert to numpy
    def finalize_set(x, y):
        if not x:
            return np.array([]), np.array([])
        return preprocess_input(np.array(x)), np.array(y)

    train_x, train_y = finalize_set(train_x, train_y)
    val_x, val_y = finalize_set(val_x, val_y)
    test_x, test_y = finalize_set(test_x, test_y)

    # Print distribution
    print("\nDataset Summary:")
    for idx, name in idx_to_class.items():
        t_c = np.sum(train_y == idx)
        v_c = np.sum(val_y == idx) if val_y.size > 0 else 0
        ts_c = np.sum(test_y == idx) if test_y.size > 0 else 0
        print(f"  Class '{name}': {t_c} train, {v_c} val, {ts_c} test")

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), idx_to_class


def get_augmentation_generator():
    """Returns an ImageDataGenerator with best-practice augmentations for faces."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


def build_model(config: Config, num_classes: int):
    """Builds a MobileNetV2-based model for transfer learning."""
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        alpha=config.MOBILE_NET_ALPHA,
    )

    # Freeze the base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.DROPOUT_RATE)(x)

    # Directly to output layer for maximum size reduction
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_metrics(model, test_x, test_y, idx_to_class):
    """Generates and saves a confusion matrix and precision-recall curve."""
    if test_x.size == 0:
        print("Warning: Skipping metrics plotting because test_x is empty.")
        return

    print("\nGenerating evaluation metrics...")

    # 1. Predictions
    y_pred_probs = model.predict(test_x)
    y_pred = np.argmax(y_pred_probs, axis=1)

    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 2. Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if not os.path.exists("images"):
        os.makedirs("images")
    plt.savefig("images/confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")

    # # 3. Precision-Recall Curve
    # plt.figure(figsize=(10, 8))
    #
    # # For each class
    # for i in range(len(classes)):
    #     # Binary true/false for this class
    #     y_true_binary = (test_y == i).astype(int)
    #     y_scores = y_pred_probs[:, i]
    #
    #     precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    #     avg_precision = average_precision_score(y_true_binary, y_scores)
    #
    #     plt.plot(recall, precision, label=f"{classes[i]} (AP={avg_precision:.2f})")
    #
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.legend(loc="best")
    # plt.savefig("precision_recall_curve.png")
    # print("Precision-Recall curve saved to precision_recall_curve.png")
    # plt.show()


def train(config: Config):
    """Orchestrates the data loading, model building, and training process."""
    (train_x, train_y), (val_x, val_y), (test_x, test_y), idx_to_class = (
        load_and_preprocess_data(config)
    )
    num_classes = len(idx_to_class)

    model = build_model(config, num_classes)
    model.summary()

    datagen = get_augmentation_generator()
    datagen.fit(train_x)

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if val_x.size > 0 else "loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss" if val_x.size > 0 else "loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\nStarting training for {num_classes} classes...")
    model.fit(
        datagen.flow(train_x, train_y, batch_size=config.BATCH_SIZE),
        steps_per_epoch=max(1, len(train_x) // config.BATCH_SIZE),
        epochs=config.EPOCHS,
        validation_data=(val_x, val_y) if val_x.size > 0 else None,
        callbacks=callbacks,
    )

    return model, idx_to_class, (train_x, train_y), (val_x, val_y), (test_x, test_y)


def export_model_to_tflite(
    model, config: Config, num_classes: int, train_x=None, gen_dir="gen"
):
    """Converts the Keras model to TFLite and exports it to C/H files."""
    print("\nExporting model to TFLite...")
    os.makedirs(gen_dir, exist_ok=True)

    # 1. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if train_x is not None:
        print("Enabling full integer quantization...")

        def representative_data_gen():
            # Use a subset of training data for calibration
            for i in range(min(100, len(train_x))):
                input_data = np.expand_dims(train_x[i], axis=0).astype(np.float32)
                yield [input_data]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to int8 (as expected by inference.cpp)
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # 2. Save .tflite file
    tflite_path = os.path.join(gen_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser(description="Train Face Recognition Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/personal/data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    config = Config(
        DATA_DIR=args.data_dir, EPOCHS=args.epochs, BATCH_SIZE=args.batch_size
    )
    if not os.path.exists("models"):
        os.makedirs("models")

    model, idx_to_class, (train_x, train_y), (val_x, val_y), (test_x, test_y) = train(
        config
    )
    num_classes = len(idx_to_class)

    # Save the class mapping for later use
    print(f"\nTraining complete. Class mapping: {idx_to_class}")

    # Plot metrics using the test set
    if test_x.size > 0:
        plot_metrics(model, test_x, test_y, idx_to_class)
    elif val_x.size > 0:
        print("No test data available, using validation data for metrics.")
        plot_metrics(model, val_x, val_y, idx_to_class)
    else:
        print("No evaluation data available to plot metrics.")
    #
    # Export to TFLite for microcontroller
    export_model_to_tflite(model, config, num_classes, train_x=train_x)


if __name__ == "__main__":
    main()
