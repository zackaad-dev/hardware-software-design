import argparse
import os

import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from main import (
    Config,
    build_model,
    get_augmentation_generator,
    load_and_preprocess_data,
)


class MyHyperModel(kt.HyperModel):
    def __init__(self, num_classes, train_x, train_y, val_x, val_y, datagen):
        self.num_classes = num_classes
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.datagen = datagen

    def build(self, hp):
        """
        Model builder function for Keras Tuner.
        Maps hyperparameters to our existing build_model logic.
        """
        config = Config(
            LEARNING_RATE=hp.Float(
                "learning_rate", min_value=1e-5, max_value=1e-2, sampling="log"
            ),
            DROPOUT_RATE=hp.Float(
                "dropout_rate", min_value=0.0, max_value=0.8, step=0.1
            ),
            DENSE_UNITS=hp.Int("dense_units", min_value=32, max_value=512, step=32),
            MOBILE_NET_ALPHA=0.35,  # Fixed as per requirements
        )

        return build_model(config, self.num_classes)

    def fit(self, hp, model, *args, **kwargs):
        """
        Custom fit method to allow tuning the batch size.
        """
        batch_size = hp.Choice("batch_size", values=[8, 16, 32, 64])

        return model.fit(
            self.datagen.flow(self.train_x, self.train_y, batch_size=batch_size),
            steps_per_epoch=max(1, len(self.train_x) // batch_size),
            validation_data=(self.val_x, self.val_y) if self.val_x.size > 0 else None,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(description="Tune Face Recognition Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/personal/data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--max-trials", type=int, default=30, help="Maximum number of trials"
    )
    args = parser.parse_args()

    config = Config(DATA_DIR=args.data_dir)

    # 1. Load Data once
    (train_x, train_y), (val_x, val_y), (test_x, test_y), idx_to_class = (
        load_and_preprocess_data(config)
    )
    num_classes = len(idx_to_class)

    # 2. Setup Data Augmentation
    datagen = get_augmentation_generator()
    datagen.fit(train_x)

    # 3. Initialize Bayesian Optimization Tuner
    hypermodel = MyHyperModel(num_classes, train_x, train_y, val_x, val_y, datagen)

    tuner = kt.BayesianOptimization(
        hypermodel,
        objective="val_accuracy" if val_x.size > 0 else "accuracy",
        max_trials=args.max_trials,
        executions_per_trial=1,
        directory="tuning_results",
        project_name="face_recognition_tuning",
        overwrite=True,
    )

    # 4. Search for optimal hyperparameters
    print("\nStarting Hyperparameter Search...")
    tuner.search(
        epochs=40,  # Shorter epochs for tuning to save time
        callbacks=[
            EarlyStopping(
                monitor="val_loss" if val_x.size > 0 else "loss",
                patience=5,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if val_x.size > 0 else "loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
            ),
        ],
    )

    # 5. Summary and Best Results
    print("\nHyperparameter Search Complete!")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest Hyperparameters Found:")
    print(f" - Learning Rate: {best_hps.get('learning_rate')}")
    print(f" - Dropout Rate: {best_hps.get('dropout_rate')}")
    print(f" - Dense Units: {best_hps.get('dense_units')}")
    print(f" - Batch Size: {best_hps.get('batch_size')}")

    # 6. Save the best model
    best_model = tuner.hypermodel.build(best_hps)
    best_model.save("models/best_tuned_model.keras")
    print("\nBest model saved to 'models/best_tuned_model.keras'")


if __name__ == "__main__":
    main()
