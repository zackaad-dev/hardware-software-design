import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.export_tflite import write_model_c_file, write_model_h_file


@dataclass
class Config:
    # Directories
    DATA_DIR: str = os.path.expanduser("~/personal/data/")
    MODEL_SAVE_PATH: str = "model.keras"

    # Image Settings
    IMG_SIZE: int = 224

    # Hyperparameters for Tuning
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.004588
    DROPOUT_RATE: float = 0.2
    DENSE_UNITS: int = 128
    MOBILE_NET_ALPHA: float = 1

    # Early Stopping
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
    for fp in glob.glob(os.path.join(folder_path, "*")):
        if fp.lower().endswith(valid_extensions):
            img_arr = load_image_as_array(fp, img_size)
            if img_arr is not None:
                x.append(img_arr)
                y.append(label)
    return x, y


def load_and_preprocess_data(config: Config):
    """
    Dynamically loads training and validation data based on folder structure.
    Expects DATA_DIR to contain subfolders for each class, e.g.:
    DATA_DIR/person_a/
    DATA_DIR/person_b/
    DATA_DIR/other/
    DATA_DIR/person_a-validation/
    ... etc.
    """
    train_x, train_y = [], []
    val_x, val_y = [], []

    # Identify classes by looking at folders that don't end in '-validation'
    all_folders = [
        f
        for f in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, f))
    ]
    classes = sorted([f for f in all_folders if not f.endswith("-validation")])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    print(f"Detected Classes: {classes}")

    for class_name in classes:
        label = class_to_idx[class_name]

        # Load training data
        train_folder = os.path.join(config.DATA_DIR, class_name)
        x, y = get_data_from_folder(train_folder, label, config.IMG_SIZE)
        train_x.extend(x)
        train_y.extend(y)

        # Load validation data
        val_folder = os.path.join(config.DATA_DIR, f"{class_name}-validation")
        if os.path.exists(val_folder):
            vx, vy = get_data_from_folder(val_folder, label, config.IMG_SIZE)
            val_x.extend(vx)
            val_y.extend(vy)
        else:
            print(f"Warning: Validation folder not found for {class_name}")

    # Convert to numpy and preprocess for MobileNetV2
    train_x = preprocess_input(np.array(train_x))
    train_y = np.array(train_y)
    val_x = preprocess_input(np.array(val_x))
    val_y = np.array(val_y)

    # Print distribution
    print("\nDataset Summary:")
    for idx, name in idx_to_class.items():
        train_count = np.sum(train_y == idx)
        val_count = np.sum(val_y == idx)
        print(f"  Class '{name}': {train_count} train, {val_count} val")

    return (train_x, train_y), (val_x, val_y), idx_to_class


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
    x = Dense(config.DENSE_UNITS, activation="relu")(x)
    x = Dropout(config.DROPOUT_RATE)(x)

    # Use softmax for multi-class classification
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_metrics(model, val_x, val_y, idx_to_class):
    """Generates and saves a confusion matrix and precision-recall curve."""
    print("\nGenerating evaluation metrics...")

    # 1. Predictions
    y_pred_probs = model.predict(val_x)
    y_pred = np.argmax(y_pred_probs, axis=1)

    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 2. Confusion Matrix
    cm = confusion_matrix(val_y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")

    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))

    # For each class
    for i in range(len(classes)):
        # Binary true/false for this class
        y_true_binary = (val_y == i).astype(int)
        y_scores = y_pred_probs[:, i]

        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
        avg_precision = average_precision_score(y_true_binary, y_scores)

        plt.plot(recall, precision, label=f"{classes[i]} (AP={avg_precision:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.savefig("precision_recall_curve.png")
    print("Precision-Recall curve saved to precision_recall_curve.png")
    plt.show()


def train(config: Config):
    """Orchestrates the data loading, model building, and training process."""
    # 1. Load Data
    (train_x, train_y), (val_x, val_y), idx_to_class = load_and_preprocess_data(config)
    num_classes = len(idx_to_class)

    # 2. Build Model
    model = build_model(config, num_classes)
    model.summary()

    # 3. Augmentation
    datagen = get_augmentation_generator()
    datagen.fit(train_x)

    # 4. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            config.MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    # 5. Fit
    print(f"\nStarting training for {num_classes} classes...")
    model.fit(
        datagen.flow(train_x, train_y, batch_size=config.BATCH_SIZE),
        steps_per_epoch=len(train_x) // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=callbacks,
    )

    return model, idx_to_class, (val_x, val_y)


def export_model_to_tflite(model, config: Config, num_classes: int, gen_dir="gen"):
    """Converts the Keras model to TFLite and exports it to C/H files."""
    print("\nExporting model to TFLite...")
    os.makedirs(gen_dir, exist_ok=True)

    # 1. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 2. Save .tflite file
    tflite_path = os.path.join(gen_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # 3. Export to C/H files
    model_h_path = os.path.join(gen_dir, "model.h")
    model_c_path = os.path.join(gen_dir, "model.c")

    defines = {
        "IMG_SIZE": config.IMG_SIZE,
        "NUM_CLASSES": num_classes,
    }
    declarations = []  # You can add declarations like 'extern float my_var;' here if needed

    write_model_h_file(model_h_path, defines, declarations)
    write_model_c_file(model_c_path, tflite_model)

    print(f"TFLite model exported to {gen_dir}/model.h and {gen_dir}/model.c")


def main():
    config = Config()
    model, idx_to_class, (val_x, val_y) = train(config)
    num_classes = len(idx_to_class)

    # Save the class mapping for later use
    print(f"\nTraining complete. Class mapping: {idx_to_class}")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Plot metrics
    if len(val_x) > 0:
        plot_metrics(model, val_x, val_y, idx_to_class)
    else:
        print("No validation data available to plot metrics.")

    # Export to TFLite for microcontroller
    export_model_to_tflite(model, config, num_classes)


if __name__ == "__main__":
    main()
