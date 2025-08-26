import tensorflow as tf
from video_utils.mask import long_term_temporal_loss_non_warp,temporal_loss, get_optimal_flow, warp_previous_frames
from Ruder.luminance import wrap_images_prior_luminance_f1
def compute_temporal_loss(combination_image, config = {}):
    frames = config.get("warp_frames", [])
    long_term = config.get("long_term", False)
    flow = config.get("flow", None)
    mask = config.get("mask", None)
    is_luminance = config.get("is_luminance ",False)
    loss = tf.constant(0.0, dtype=tf.float32)
    
    if flow is None:
        print("ERROR: flow is None, returning zero loss")
        return loss
    if len(frames) > 0:
        if long_term:
            temporal_error = long_term_temporal_loss_non_warp(
                curr_stylized_frame=combination_image, previous_warp_frames=frames, mask=mask
            )
            loss += temporal_error
        else:
            # Since items are appended to frames, the previous frame is the last element
            prev_frame = frames[-1]
            temporal_error = temporal_loss(
                prev_frame, combination_image, mask=mask
            )
            loss += temporal_error
        if is_luminance:
            prev_frame = frames[-1]
            non_warp_frames = config.get("frames",[])
            curr_img = config.get("curr_img", None)
            prev_non_warped_frame = non_warp_frames[-1]
            wrap_images_prior_luminance_f1(prev_non_warped_frame,curr_img,prev_frame,combination_image,mask,flow)
            
    return loss
                     

                    
