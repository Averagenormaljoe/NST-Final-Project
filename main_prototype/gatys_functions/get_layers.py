# Now define the style and content layers for the project along with the weights for each layer.
from shared_utils.gatys_network import get_content_layer_names, get_style_layer_names, get_style_weights, get_content_weights


def get_layers(use_custom: bool,loss_network : str) -> tuple[list[str], list[str], dict, dict]:
    # decides if custom layers and weights should be used
    if use_custom:
        style_layer_names = get_style_layer_names(loss_network)
        content_layer_names = get_content_layer_names(loss_network)
        style_weights = get_style_weights(loss_network)
        content_weights = get_content_weights(loss_network)
    else:
        style_layer_names, content_layer_names, style_weights, content_weights = get_default_NST_layers()
    return style_layer_names, content_layer_names, style_weights, content_weights


def get_default_NST_layers() -> tuple[list[str], list[str], dict, dict]:
    # style layers
    style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
    ]
    # content layers
    content_layer_names = ["block5_conv2"]
    # weights for the style layers
    style_weights = {'block1_conv1': 1.,
                    'block2_conv1': 0.8,
                    'block3_conv1': 0.5,
                    'block4_conv1': 0.3,
                    'block5_conv1': 0.1}
    # weights for the content layers
    content_weights = {'block5_conv2': 1e-6}
    
    return style_layer_names, content_layer_names, style_weights, content_weights