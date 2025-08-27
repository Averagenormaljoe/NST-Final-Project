import tensorflow as tf
from video_utils.mask import long_term_temporal_loss_non_warp,temporal_loss, get_optimal_flow, warp_previous_frames
from Ruder.luminance import  no_luminance, wrap_images_prior_luminance
def compute_temporal_loss(combination_image, config = {}):
    long_term = config.get("long_term", False)
    is_flow = config.get("flow", True)
    is_mask =  config.get("mask", True) 
    frames = config.get("warp_frames", []) if is_flow else config.get("frames",[])
    flow = config.get("flow", None) if is_flow else None
    mask = config.get("mask", None) if is_mask else None
    temporal_weight =  config.get("temporal_weight", 1.0)
    is_luminance = config.get("is_luminance ",False)
    luminance_version = config.get("is_luminance ",1)
    loss = tf.constant(0.0, dtype=tf.float32)
    if config is None:
        raise ValueError("Error: config cannot be none.")
    if isinstance(config,dict):
        raise ValueError(f"Error: config is not a dictionary instead ({type(config)}).")
    if flow is None:
        print("ERROR: flow is None, returning zero loss")
        return loss
    if combination_image is None:
        print("ERROR: combination_image is None, returning zero loss")
        return loss
    if len(frames) > 0:
        if long_term:
            temporal_error = long_term_temporal_loss_non_warp(
                curr_stylized_frame=combination_image, previous_warp_frames=frames, mask=mask
            )
            loss += temporal_error * temporal_weight
        else:
            # Since items are appended to frames, the previous frame is the last element
            prev_frame = frames[-1]
            temporal_error = temporal_loss(
                prev_frame, combination_image, mask=mask
            )
            loss += temporal_error * temporal_weight
        if is_luminance:
            prev_frame = frames[-1]
            non_warp_frames = config.get("frames",[])
            curr_img = config.get("base_img", None)
            prev_non_warped_frame = non_warp_frames[-1]
            if luminance_version == 3:
                luminance_loss = no_luminance(prev_frame,combination_image,mask)
            else:
                luminance_weight =  config.get("luminance_weight", 1.0)
                luminance_type = "f2" if luminance_version == 2 else "f1"
                luminance_loss = wrap_images_prior_luminance(prev_non_warped_frame,curr_img,prev_frame,combination_image,mask,flow,luminance_type)
            loss += luminance_loss * luminance_weight
            
    return loss
                     

                    
