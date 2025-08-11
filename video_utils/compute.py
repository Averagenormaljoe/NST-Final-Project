import tensorflow as tf
from video_utils.mask import long_term_temporal_loss_non_warp,temporal_loss, get_optimal_flow, warp_previous_frames
def compute_temporal_loss(combination_image, config = {}):
    frames = config.get("warp_frames", [])
    long_term = config.get("long_term", False)
    flow = config.get("flow", None)
    loss = tf.zeros(shape=()) 
    
    if flow is None:
        print("ERROR: flow is None, returning zero loss")
        return loss
    if len(frames) > 0:
        if long_term:
            temporal_error = long_term_temporal_loss_non_warp(
                curr_stylized_frame=combination_image, previous_warp_frames=frames 
            )
            loss += temporal_error
        else:
            # Since items are appended to frames, the previous frame is the last element
            prev_frame = frames[-1]
            temporal_error = temporal_loss(
                prev_frame, combination_image
            )
            loss += temporal_error

    return loss
                     

                    
