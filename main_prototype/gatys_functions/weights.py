from shared_utils.gatys_network import get_content_layer_names, get_style_layer_names
def get_loss_layers(config):
    loss_network = config.get("loss_network","vgg19")
    style_names = config.get("style_layer_names", get_style_layer_names(loss_network))
    content_names = config.get("content_layer_names", get_content_layer_names(loss_network))
    return content_names, style_names
def get_weights(config):
    content_weight = config.get("c_weight", 2.5e-8)
    style_weight = config.get("s_weight", 1e-6)
    total_variation_weight = config.get("tv_weight", 1e-6)
    return content_weight, style_weight, total_variation_weight
def compute_loss(combination_image, base_image, style_reference_image, config = {}):