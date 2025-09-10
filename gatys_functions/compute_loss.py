import tensorflow as tf
from gatys_functions.weights import get_loss_layers, get_weights
from shared_utils.loss_functions import content_loss, style_loss, total_variation_loss
from gatys_functions.get_features import get_features
def compute_loss(combination_image : tf.Tensor, base_image : tf.Tensor, style_reference_image : tf.Tensor, config : dict = {}) -> tuple[tf.Tensor,dict]:
    metrics_dict : dict = {}
    tv_type : str = config.get("tv_type", "gatys")
    size : tuple[int,int] = config.get("size", (400, 400))
    feature_extractor = config.get("feature", None)
    content_weight, style_weight, total_variation_weight = get_weights(config)
    if feature_extractor is None:
      print("Error: feature extractor has not be provided.")
      return None, metrics_dict
    content_names, style_names = get_loss_layers(config)
    input_tensor = tf.concat(
    [base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)
    loss = tf.zeros(shape=())
    w,h = size
    content_weight_per_layer : float = content_weight / len(content_names)
    c_loss = tf.zeros(shape=())
    # content layer iteration
    for layer_name in content_names:
      base_features, combination_features = get_features(features, layer_name, base_index=0, combination_index=2)
      c_loss += content_weight_per_layer  * content_loss(
          base_features, combination_features
      )
    loss += c_loss
    metrics_dict["content"] =  float(c_loss)
    s_loss = tf.zeros(shape=())
    style_weight_per_layer : float = style_weight / len(style_names)
    # style layer iteration
    for layer_name in style_names:
      base_features, combination_features = get_features(features, layer_name, base_index=1, combination_index=2)
      style_loss_value = style_loss(
      base_features, combination_features, w, h)
      s_loss +=  style_weight_per_layer * style_loss_value

    loss += s_loss
    metrics_dict["style"] =  float(s_loss)
    is_tv = config.get("is_tv", True)
    if is_tv:
      # calculate the total variation loss
      t_loss = total_variation_weight * total_variation_loss(combination_image,tv_type=tv_type, size=size)
      loss += t_loss
      metrics_dict["total_variation"] = float(t_loss)
    return loss, metrics_dict