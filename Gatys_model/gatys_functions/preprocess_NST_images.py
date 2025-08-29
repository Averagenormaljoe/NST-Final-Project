import tensorflow as tf
from Gatys_model.helper_functions.device_helper import get_device
from Gatys_model.helper_functions.helper import match_style_color_to_base, preprocess_image
from Gatys_model.helper_functions.image_helper import add_noise_to_image
def preprocess_NST_images(base_image_path : str, style_reference_image_path : str, config : dict = {},device_config : dict = {}):
    size = config.get("size",(400,400))
    w,h = size
    GPU_in_use = device_config.get("gpu",0)
    CPU_in_use = device_config.get("cpu",0)
    preserve_color = config.get("preserve_color",False)
    noise = config.get("noise",False)
    with get_device(GPU_in_use, CPU_in_use):
        base_image = preprocess_image(base_image_path,w,h)
        style_reference_images = preprocess_image(style_reference_image_path, w,h)
        if preserve_color:
            style_reference_images = match_style_color_to_base(base_image, style_reference_images)
        if noise:
            initial_combination_image = add_noise_to_image(base_image)
            combination_image = tf.Variable(initial_combination_image)
        else:
            combination_image = tf.Variable(preprocess_image(base_image_path,w,h))
    return base_image, style_reference_images, combination_image
