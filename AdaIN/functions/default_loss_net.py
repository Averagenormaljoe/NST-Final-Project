import keras


def default_loss_net(image_size: tuple[int, int]):
    model = keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(*image_size, 3),
        )
    layer_names: list[str] = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    return model, layer_names