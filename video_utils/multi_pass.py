# multi pass algorithm adapted from 'https://arxiv.org/abs/1604.08610' paper by Ruder et al.
from time import time
import numpy as np
from video_utils.mask import warp_flow
from tqdm import trange
import traceback
def get_pass_range(direction : str,frames_length : int):
    initial_range = range(frames_length)
    range_fn = initial_range if direction == "f" else reversed(initial_range)
    return range_fn


def multi_pass(n_pass : int,flows : list,style_image : str,masks: list, blend_weight : float =0.5,temporal_loss_after_n_passes : int = 3,config : dict = {}):
    combination_frames = config.get("frames",[])


    if not isinstance(flows, list):
        raise TypeError(f"flows is not a list ({type(flows)}).")
    if not isinstance(masks, list):
        raise TypeError(f"masks is not a list({type(masks)}).")
    if not isinstance(n_pass,int):
        print(f"Error: n_pass is not an int ({type(n_pass)}).")
    if not isinstance(temporal_loss_after_n_passes,int):
        print(f"Error: temporal_loss_after_n_passes is not an int ({type(temporal_loss_after_n_passes)}).")
    if not isinstance(blend_weight,float):
        raise TypeError(f"Error: blend_weight is not a float ({type(blend_weight)}).")
    if n_pass <= 0:
            raise ValueError(f"n_pass is not a positive integer ({n_pass}).")
    if temporal_loss_after_n_passes <= 0:
            raise ValueError(f"temporal_loss_after_n_passes is not a positive integer ({n_pass}).")    
    if not isinstance(style_image,str):
       raise TypeError(f"Error: style_image is not a str ({type(style_image)}).")  
    frames_length = len(combination_frames)
    flow_length = len(flows)
    mask_length = len(masks)
    if flow_length != frames_length - 1:
        raise ValueError(f"Length of the  flows is incorrect. Flow length: {flow_length}, Frames length: {frames_length}")
    if mask_length != frames_length - 1:
        raise ValueError(f"Length of the masks is incorrect. Mask length: {mask_length}, Frames length: {frames_length}")
        
    if frames_length == 0:
        print("Error: frames length is zero.")
        raise ValueError("Error: frame list is empty")
    if flow_length == 0:
        print("Error: flow length is zero.")
        raise ValueError("Error: flow list is empty")
    if mask_length == 0:
        print("Error: mask length is zero.")
        raise ValueError("Error: mask list is empty")
    if not blend_weight <= 1 or not blend_weight >= 0:
            raise ValueError(f"Error: blend_weight is not between 0 and 1 ({blend_weight}).")
    start : float = time()
    pass_time : list = []
    stylize_frames = combination_frames.copy()

    neg_blend_weight = 1 - blend_weight

    for j in trange(0, n_pass, desc=f"Processing passes in multi pass algorithm"):
        pass_tick : float = time()
        direction : str = "f" if j % 2 == 0 else "b"
        pass_range : range = get_pass_range(direction,flow_length)
        prev_img = None
        is_temporal_loss : bool = j >= temporal_loss_after_n_passes
        config["video_mode"] = is_temporal_loss
        for i in pass_range:
            try:
                if (direction == "f" and i == 0) or (direction == "b" and i == mask_length - 1):
                    prev_img = stylize_frames[i]
                    stylize_frames[i] = prev_img
                else:
                    index_d : int = i - 1 if direction == "f" else i + 1 
                    if index_d < 0 or index_d >= mask_length:
                        print(f"index_d ({index_d}) is invalid. Failed to access frames. Length: {mask_length}")
                        continue
                    print(index_d)
                    print(len(masks))
                    warp_mask = masks[index_d]
                    next_img = combination_frames[index_d]
                    reverse_flow = True if direction == "b" else False
                    try:
                        warp_img = warp_flow(next_img,flows[index_d],reverse_flow)
                    except Exception as e:
                        prev_img = stylize_frames[i]
                        traceback.print_exc()
                        print(f"Error with warp_flow. {index_d} for pass {j}, Flow length: {flow_length} Message: {e}")
                    first_mul = blend_weight * warp_mask * warp_img
                    ones_res = np.ones_like(warp_mask, dtype=np.float32)
                    neg_prev_mask = ones_res - warp_mask
                    second_mul = ((neg_blend_weight * ones_res) + (blend_weight * neg_prev_mask)) * prev_img
                    final_result = first_mul + second_mul
                    stylize_frames[i] = final_result
                    prev_img = stylize_frames[i] 
            except Exception as e:
                traceback.print_exc()
                print(f"Error: during frame {i} for pass {j}, Length: {mask_length} Message: {e}")
                prev_img = stylize_frames[i]
        pass_end : float = time()
        duration : float = pass_end - pass_tick
        pass_time.append(duration)
    end : float = time()
    total_pass_duration : float = end - start
    print(f"Multi-pass process ({total_pass_duration :.2f}) seconds")
    return stylize_frames

