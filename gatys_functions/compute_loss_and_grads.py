import tensorflow as tf
from gatys_functions.compute_loss import compute_loss
from helper_functions.device_helper import get_device
from shared_utils.exception_checks import none_check
from video_utils.compute import compute_temporal_loss


@tf.function
def compute_loss_and_grads(combination_image : tf.Tensor,  base_image : tf.Tensor, style_images : tf.Tensor | list, config : dict= {},device_config : dict = {}):
    none_check(combination_image, "combination_image")
    none_check(base_image, "base_image")
    none_check(style_images, "style_images")
    none_check(config, "config")
    none_check(device_config, "device_config")
    
    verbose = config.get("verbose", 0)
    if verbose > 0:
        tf.print("combination_image == tf.Variable", isinstance(combination_image, tf.Variable))
    # ensures that 'style_images' is a list
    type_style_images : list = style_images if isinstance(style_images,list) else [style_images]
    style_weight : float = config.get("s_weight", 1e-6)
    # get the device for computation
    all_metrics : list = []
    # for storing losses that are not style or content
    metrics_dict = {}
    GPU_in_use : int = device_config.get("gpu", 0)
    CPU_in_use : int = device_config.get("cpu", 0)
    video_mode : bool = config.get("video_mode",True)

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
            if video_mode:
                t_loss = compute_temporal_loss(combination_image, config)
                
                if t_loss > 0:
                    loss += t_loss
                    metrics_dict["temporal_loss"] = t_loss
                else:
                    metrics_dict["temporal_loss"] = tf.constant(0.0, dtype=tf.float32)
        
       
        grads = tape.gradient(loss, combination_image)
        return loss, grads, all_metrics , metrics_dict