import tensorflow as tf
from AdaIN_functions.ada_in import ada_in
from shared_utils.get_custom_metrics import get_custom_metrics
def stylize_image(model, content_image : tf.Tensor, style_image : tf.Tensor):
    """Stylizes the content image using the style image."""
    content_encoded = model.encoder(content_image)
    style_encoded = model.encoder(style_image)
    t = ada_in(style=style_encoded, content=content_encoded)
    
    reconstructed_image = model.decoder(t)
    return reconstructed_image

def get_image_metrics(model,content_image : tf.Tensor,style_image : tf.Tensor):
    reconstructed_image = stylize_image(model, content_image, style_image)
    all_metrics = get_custom_metrics(content_image, reconstructed_image)
    return  reconstructed_image, all_metrics