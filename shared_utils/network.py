import keras


def get_model_for_loss_net(name : str,image_size: tuple[int, int] = (224, 224))-> keras.Model:
    if name == "vgg19":
        vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return vgg19
 
    elif name == "vgg16":
        vgg16 = keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return vgg16
    else:
        vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return vgg19