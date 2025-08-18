import tensorflow as tf

from shared_utils.gatys_network import get_content_layer_names, get_style_layer_names
from shared_utils.loss_functions import content_loss, style_loss, total_variation_loss


def get_loss_layers(config):
    loss_network = config.get("loss_network","vgg19")
    style_names = config.get("style_layer_names", get_style_layer_names(loss_network))
    content_names = config.get("content_layer_names", get_content_layer_names(loss_network))
    return content_names, style_names
def get_single_weights(config):
    content_weight = config.get("c_weight", 1.0)
    style_weight = config.get("s_weight", 1.0)
    total_variation_weight = config.get("tv_weight", 1e-4)
    return content_weight, style_weight, total_variation_weight
def compute_loss(combination_image, base_image, style_reference_image, config = {}):
  metrics_dict = {}
  tv_type = config.get("tv_type", "gatys")
  size = config.get("size", (400, 400))
  feature_extractor = config.get("feature", None)
  if feature_extractor is None:
    print("Error: feature extract has not be provided.")
    return None, metrics_dict
  content_names, style_names = get_loss_layers(config)
  input_tensor = tf.concat(
  [base_image, style_reference_image, combination_image], axis=0)
  features = feature_extractor(input_tensor)
  loss = tf.zeros(shape=())
  w,h = size
  content_weight_per_layer : float = single_content_weight / len(content_names)
  c_loss = tf.zeros(shape=())
  # content layer iteration
  for layer_name in content_names:
    layer_features = features[layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    c_loss += content_weight_per_layer  * content_loss(
        base_image_features, combination_features
    )
  loss += c_loss
  metrics_dict["content"] =  float(c_loss)
  s_loss = tf.zeros(shape=())
  style_weight_per_layer : float = single_style_weight / len(style_names)
  # style layer iteration
  for layer_name in style_names:
    layer_features = features[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss_value = style_loss(
    style_reference_features, combination_features, w, h)
    s_loss +=  style_weight_per_layer * style_loss_value

  loss += s_loss
  metrics_dict["style"] =  float(s_loss)
  # calculate the total variation loss
  t_loss = total_variation_weight * total_variation_loss(combination_image,tv_type=tv_type, size=size)
  loss += t_loss
  metrics_dict["total_variation"] = float(t_loss)
  return loss, metrics_dict