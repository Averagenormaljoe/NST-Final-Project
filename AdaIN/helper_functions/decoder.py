# This is the same as the decoder but with normalization layers added. This is said to hurt the performance of the model,
# which is why it is not used in the final model but we would like to test it for hyperparameter purposes.
def get_normalization_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),

            layers.Conv2D(filters=512, **config),
            layers.BatchNormalization(),

            layers.UpSampling2D(),

            layers.Conv2D(filters=256, **config),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, **config),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, **config),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, **config),
            layers.BatchNormalization(),

            layers.UpSampling2D(),

            layers.Conv2D(filters=128, **config),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, **config),
            layers.BatchNormalization(),

            layers.UpSampling2D(),

            layers.Conv2D(filters=64, **config),
            layers.BatchNormalization(),

            layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="tanh"  # Final layer – no BatchNorm here
            ),
        ]
    )
    return decoder
