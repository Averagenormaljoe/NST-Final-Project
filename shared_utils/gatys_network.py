
def get_content_layer_names(name: str = "vgg19") -> list[str]:
    if name == "vgg19":
        return ["block5_conv2"]
    return ["block5_conv2"]
def get_style_layer_names(name: str = "vgg19") -> list[str]:
    if name == "vgg19":
        return ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv2"]
    else:
        return ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv2"]

def get_style_weights(name: str = "vgg19"):
    if name == "vgg19":
        style_weights = {'block1_conv1': 1.,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}
  
        return style_weights
    else:
        style_weights = {'block1_conv1': 1.,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}
  
        return style_weights
def get_content_weights(name: str = "vgg19"):
    if name == "vgg19":
        return {'block5_conv2': 1.0}
    else:
        return {'block5_conv2': 1.0}