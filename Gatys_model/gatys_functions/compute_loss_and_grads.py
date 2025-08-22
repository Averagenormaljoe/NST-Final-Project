import tensorflow as tf
from gatys_functions.compute_loss import compute_loss
from helper_functions.device_helper import get_device
from video_utils.compute import compute_temporal_loss

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_images, config= {},device_config = {}):
    verbose = config.get("verbose", 0)
    if verbose > 0:
        tf.print("combination_image == tf.Variable", isinstance(combination_image, tf.Variable))
    # ensures that 'style_images' is a list
    type_style_images = style_images if isinstance(style_images,list) else [style_images]
    style_weight = config.get("s_weight", 1e-6)
    # get the device for computation
    all_metrics = []
    # for storing losses that are not style or content
    metrics_dict = {}
    GPU_in_use = device_config.get("gpu", 0)
    CPU_in_use = device_config.get("cpu", 0)
    style_weight = config.get("s_weight", 1e-6)
    with get_device(GPU_in_use, CPU_in_use):  
        with tf.GradientTape() as tape:
            loss = tf.zeros(shape=())
            num : int = len(type_style_images)
            style_cal = style_weight / num
            # iterate through the style images
            for image in type_style_images:
                style_loss_value, style_metrics = compute_loss(
                    combination_image, base_image, image,config  
                )
                loss += style_loss_value
                all_metrics.append(style_metrics)
            t_loss = compute_temporal_loss(combination_image, config)
            
            if t_loss > 0:
                loss += t_loss
                metrics_dict["temporal_loss"] = t_loss
            else:
                metrics_dict["temporal_loss"] = tf.constant(0.0, dtype=tf.float32)
        grads = tape.gradient(loss, combination_image)
        return loss, grads, all_metrics , metrics_dict