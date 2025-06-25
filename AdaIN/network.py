
import tensorflow 
from tensorflow import keras
def get_layer_names_for_loss_net(name: str) -> list[str]:
    if name == "vgg19":
        layer_names : list[str] = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv2", "block5_conv1"]
        return layer_names
    elif name == "altvgg19":
        layer_names : list[str] =  ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"]
        return layer_names
    elif name == "mobilenet":
        layer_names : list[str] = ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]
        return layer_names
    elif name == "efficientnet":
        layer_names : list[str] = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]
        return layer_names
    elif name == "resnet50":
        layer_names : list[str] = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
        return layer_names
    elif name == "inceptionv3":
        layer_names : list[str] = ["conv2d_1", "conv2d_2", "conv2d_3", "mixed7"]
        return layer_names
    elif name == "xception":
        layer_names : list[str] = ["block1_conv1", "block2_sepconv2_bn", "block3_sepconv2_bn", "block4_sepconv2_bn"]
        return layer_names



def get_model_for_loss_net(name,image_size: tuple[int, int] = (224, 224))-> keras.Model:
    if name == "vgg19":
        vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return vgg19
    elif name == "mobilenet":
        mobilenet = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return mobilenet
    elif name == "efficientnet":
        efficientnet = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return efficientnet
    elif name == "resnet50":
        resnet50 = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return resnet50
    elif name == "inceptionv3":
        inceptionv3 = keras.applications.InceptionV3(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return inceptionv3
    elif name == "xception":
        xception = keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return xception
    elif name == "densenet121":
        densenet121 = keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
        )
        return densenet121