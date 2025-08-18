from main_prototype.gatys_functions.get_model import get_model
from shared_utils.gatys_network import get_content_layer_names, get_content_weights, get_style_layer_names, get_style_weights


def update_model(loss_network: str, size : tuple[int,int]):
    w,h = size
    feature_extractor = get_model(loss_network, w,h)
    style_layer_names = get_style_layer_names(loss_network)
    content_layer_names = get_content_layer_names(loss_network)
    style_weights = get_style_weights(loss_network)
    content_weights = get_content_weights(loss_network)
    return feature_extractor, style_layer_names, content_layer_names, style_weights, content_weights