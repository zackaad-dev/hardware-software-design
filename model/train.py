## Barebones training script for a simple image classification model using MobileNetV2 as a base.

import tensorflow as tf
from tensorflow.keras import layers, models, applications

train_ds = tf.keras.utils.image_dataset_from_directory(
    "../data/",
    image_size=(240, 240),
    batch_size=30,
    label_mode="binary"
)


base_model = applications.MobileNetV2(input_shape=(240, 240, 3), include_top=False, alpha=0.35)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(240, 240, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=5)

model.save("simple_mode.keras")
print("Model saved to simple_mode.keras")