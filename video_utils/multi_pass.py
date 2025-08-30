# multi pass algorithm adapted from 'https://arxiv.org/abs/1604.08610' paper by Ruder et al.
from time import time
import tensorflow as tf
from Gatys_model.gatys_functions.LoopManager import LoopManager
from video_utils.mask import warp_flow
from tqdm import trange

def get_pass_range(direction : str,frames_length : int):
    initial_range = range(frames_length)
    range_fn = initial_range if direction == "f" else reversed(initial_range)
    return range_fn


def multi_pass(n_pass : int,flows : list,style_image : tf.Tensor,masks: list, blend_weight : float =0.5,temporal_loss_after_n_passes : int = 3,config : dict = {}):
    combination_frames = config.get("frames",[])
    
    if not isinstance(flows, list):
        raise TypeError(f"flows is not a list ({type(flows)}).")
    if not isinstance(masks, list) or not all(isinstance(m, tf.Tensor) for m in masks):
        raise TypeError(f"masks is not a list({type(masks)}).")
    if not isinstance(n_pass,int):
        print(f"Error: n_pass is not an int ({type(n_pass)}).")
    if n_pass <= 0:
            raise ValueError(f"n_pass is not a positive integer ({n_pass}).")
    if not isinstance(style_image,tf.Tensor):
       raise TypeError(f"Error: style_image is not a tensor ({type(style_image)}).")  
    frames_length = len(combination_frames)
    if frames_length == 0:
        print("Error: frames length is zero.")
        raise ValueError("Error: frames list is empty")
    if not  blend_weight <= 1 and not blend_weight >= 0:
            raise ValueError(f"Error: blend_weight is not between 0 and 1 ({blend_weight}).")
    start : float = time()
    pass_time : list = []
    stylize_frames = combination_frames.copy()

    neg_blend_weight = 1 - blend_weight
    loop_manager = LoopManager(config)
    for j in trange(0, n_pass, desc=f"Processing passes in multi pass algorithm"):
        pass_tick : float = time()
        direction : str = "f" if j % 2 == 0 else "b"
        pass_range : range = get_pass_range(direction,frames_length)
        prev_img = None
        is_temporal_loss : bool = j >= temporal_loss_after_n_passes
        config["video_mode"] = is_temporal_loss
        for i in pass_range:
            try:
                index_d : int = i - 1 if direction == "f" else i + 1
                if (direction == "f" and i == 0) or (direction == "b" and i == frames_length - 1):
                    prev_img = stylize_frames[i]
                    stylize_frames[i] = prev_img
                    
                else:
                    warp_mask = masks[index_d]
                    next_img = combination_frames[index_d]
                    reverse_flow = True if direction == "b" else False
                    warp_img = warp_flow(next_img,flows[index_d],reverse_flow)
                    first_mul = blend_weight * warp_mask * warp_img
                    ones_res = tf.ones_like(warp_mask)
                    neg_prev_mask = tf.subtract(ones_res, warp_mask) 
                    second_mul = ((neg_blend_weight * ones_res) + (blend_weight * neg_prev_mask)) * prev_img
                    final_result = tf.add(first_mul,second_mul)
                    config["combination_frame"] = final_result
                    generated_frames, best_frame, log_data = loop_manager.training_loop(content_path=combination_frames[i],  style_path=style_image,config=config)
                    if not generated_frames or not best_frame or not log_data:
                        print("Error: optimization loop failed. Skipping")
                        stylize_frames[i] = final_result
                        prev_img = stylize_frames[i] 
                        continue
                    stylize_frames[i] = best_frame.get_image()
                    prev_img = stylize_frames[i]
            except Exception as e:
                print(f"Error: during frame {i} for pass {j}: {e}")
                prev_img = stylize_frames[i]
        pass_end : float = time()
        duration : float = pass_end - pass_tick
        pass_time.append(duration)
    end : float = time()
    total_pass_duration : float = end - start
    print(f"Multi-pass process ({total_pass_duration :.2f}) seconds")
    return stylize_frames

