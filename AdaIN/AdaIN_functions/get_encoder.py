import keras
from keras import layers
IMAGE_SIZE = (224, 224)
def get_encoder(image_size: tuple[int, int] = IMAGE_SIZE,custom_layers : bool = False) -> keras.Model:
    vgg19 = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    vgg19.trainable = False
    if custom_layers:
        layer_names : list[str] = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
        outputs = [vgg19.get_layer(name).output for name in layer_names]
        mini_vgg19 = keras.Model(vgg19.input, outputs)
    else:
        mini_vgg19 = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = layers.Input([*image_size, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")
