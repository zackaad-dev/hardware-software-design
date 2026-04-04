import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from main import (
    Config,
    build_model,
    get_augmentation_generator,
    load_and_preprocess_data,
)

# Global variable to store num_classes for the tuner's model builder
NUM_CLASSES = None


def build_tuning_model(hp):
    """
    Model builder function for Keras Tuner.
    Maps hyperparameters to our existing build_model logic.
    """
    config = Config(
        LEARNING_RATE=hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        ),
        DROPOUT_RATE=hp.Float("dropout_rate", min_value=0.1, max_value=0.6, step=0.1),
        DENSE_UNITS=hp.Int("dense_units", min_value=64, max_value=256, step=64),
        MOBILE_NET_ALPHA=hp.Choice("mobile_net_alpha", values=[0.35, 0.5, 0.75, 1.0]),
    )

    return build_model(config, NUM_CLASSES)


def main():
    global NUM_CLASSES
    config = Config()

    # 1. Load Data once
    (train_x, train_y), (val_x, val_y), idx_to_class = load_and_preprocess_data(config)
    NUM_CLASSES = len(idx_to_class)

    # 2. Setup Data Augmentation
    datagen = get_augmentation_generator()
    datagen.fit(train_x)

    # 3. Initialize Bayesian Optimization Tuner
    # We use 'val_accuracy' as the objective to maximize
    tuner = kt.BayesianOptimization(
        build_tuning_model,
        objective="val_accuracy",
        max_trials=15,  # Increase this for more thorough tuning
        executions_per_trial=1,
        directory="tuning_results",
        project_name="face_recognition_tuning",
        overwrite=True,
    )

    # 4. Search for optimal hyperparameters
    print("\nStarting Hyperparameter Search...")
    tuner.search(
        datagen.flow(train_x, train_y, batch_size=config.BATCH_SIZE),
        steps_per_epoch=len(train_x) // config.BATCH_SIZE,
        epochs=20,  # Shorter epochs for tuning to save time
        validation_data=(val_x, val_y),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
        ],
    )

    # 5. Summary and Best Results
    print("\nHyperparameter Search Complete!")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters Found:")
    print(f" - Learning Rate: {best_hps.get('learning_rate')}")
    print(f" - Dropout Rate: {best_hps.get('dropout_rate')}")
    print(f" - Dense Units: {best_hps.get('dense_units')}")
    print(f" - MobileNet Alpha: {best_hps.get('mobile_net_alpha')}")

    # 6. Optional: Save the best model
    best_model = tuner.hypermodel.build(best_hps)
    best_model.save("best_tuned_model.keras")
    print("\nBest model saved to 'best_tuned_model.keras'")


if __name__ == "__main__":
    main()
