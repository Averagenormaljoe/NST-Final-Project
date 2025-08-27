from time import time
import cv2
import tensorflow as tf
from tqdm import trange
from Gatys_model.gatys_functions.LoopManager import LoopManager
from shared_utils.losses import temporal_loss
import numpy as np
import keras_hub
def generate_mask(img,segmenter):
  
    mask = segmenter.predict(img)
    return mask
def get_mask(img):
    segmenter = get_segmenter()
    mask = generate_mask(img, segmenter)
    return mask

def get_simple_mask(img, flow, reverse=False):
    img_shape = img.shape
    mask_of_ones = np.ones(img_shape)
    warped_mask = warp_flow(mask_of_ones, flow, reverse=reverse)
    filter_value = 0.9999
    filtered_mask = (warped_mask > filter_value)
    float_mask = filtered_mask.astype(np.float32)
    return float_mask        
    
def get_segmenter(url =  "deeplab_v3_plus_resnet50_pascalvoc"):
    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        url,  num_classes=2
        )
    return segmenter

def convert_to_numpy(img):
    if hasattr(img, 'numpy'):
        return img.numpy()
    return img


def get_optimal_flow(prev_img, curr_img, config={}):
    prev_numpy_img = convert_to_numpy(prev_img)
    curr_numpy_img = convert_to_numpy(curr_img)
    flow = load_optical_flow(prev_numpy_img, curr_numpy_img, config)
    return flow
    
def convert_to_grayscale(img):
    if len(img.shape) == 3:
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img   
    
def load_optical_flow(prev_numpy_img, curr_numpy_img, config): 
    prev_img = convert_to_grayscale(prev_numpy_img)
    curr_img = convert_to_grayscale(curr_numpy_img)
    flow_type = config.get("flow_type", "farneback")
    save_flow = config.get("save_flow", False)
    output_path = config.get("flow_output_path", "optical_flow.flo")
    if flow_type == 'deepflow':
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(prev_img, curr_img, None)
    elif flow_type == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif flow_type == 'lucas_kanade':
        prev_pts = cv2.goodFeaturesToTrack(prev_img, maxCorners=100, qualityLevel=0.3, minDistance=7)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
        flow = (prev_pts, curr_pts, status)
    elif flow_type == 'dis':
        dis = cv2.optflow.createOptFlow_DIS()
        flow = dis.calc(prev_img, curr_img, None)
     
    if save_flow:
        cv2.optflow.writeOpticalFlow(output_path, flow)
    return flow

def warp_flow(img, flow,reverse=False):
    cast_flow = flow.astype(np.float32)
    h, w = cast_flow.shape[:2]
    if reverse:
        cast_flow = -cast_flow
    cast_flow_copy = np.copy(cast_flow)
    cast_flow_copy[:,:,0] += np.arange(w)
    cast_flow_copy[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, cast_flow_copy, None, cv2.INTER_LINEAR)
    return res

def generate_occlusion_masks(prev_frame,curr_frame):
    pass

def feature_map_temporal_loss(prev_feature_map, curr_feature_map,flow, mask=None):
    optimal_flow = warp_flow(prev_feature_map,flow) 
    tl = temporal_loss(optimal_flow, curr_feature_map, mask=mask)
    return tl

def temporal_warping_error(prev_stylized_frame, curr_stylized_frame, flow, mask=None):
    warped_prev = warp_flow(prev_stylized_frame, flow)
    twe = temporal_loss(warped_prev, curr_stylized_frame, mask=mask)
    return twe
def warp_previous_frames(previous_stylized_frames, flow):
    warped_frames = []
    for prev_frame in previous_stylized_frames:
        warped_frame = warp_flow(prev_frame, flow)
        warped_frames.append(warped_frame)
    return warped_frames
def long_term_temporal_loss(curr_stylized_frame, flow, mask=None,previous_stylized_frames=[]):    
    loss = tf.zeros(shape=())
    for prev_frame in previous_stylized_frames:

        twe = temporal_warping_error(prev_frame, curr_stylized_frame, flow, mask=mask)
        loss += twe
    return loss

def long_term_temporal_loss_non_warp(curr_stylized_frame, mask=None,previous_warp_frames=[],):    
    loss = tf.zeros(shape=())
    for prev_frame in previous_warp_frames:
        twe = temporal_loss(prev_frame, curr_stylized_frame, mask=mask)
        loss += twe
    return loss
def get_pass_range(direction : str,frames_length : int):
    initial_range = range(frames_length)
    range_fn = initial_range if direction == "f" else reversed(initial_range)
    return range_fn
# multi pass algorithm adapted from 'https://arxiv.org/abs/1604.08610' paper by Ruder et al.
def multi_pass(n_pass : int,flows : list,style_image : tf.Tensor,masks: list, blend_weight : float =0.5,temporal_loss_after_n_passes : int = 3,config : dict = {}):
    combination_frames = config.get("frames",[])
    if not isinstance(n_pass,int):
        print(f"Error: n_pass is not an int ({type(n_pass)}).")
    if not isinstance(style_image,tf.Tensor):
        print(f"Error: style_image is not a tensor ({type(style_image)}).")  
    start : float = time()
    pass_time : list = []
    stylize_frames = combination_frames.copy()
    frames_length = len(combination_frames)
    neg_blend_weight = 1 - blend_weight
    loop_manager = LoopManager(config)
    for j in trange(0, n_pass, desc=f"Processing passes in multi pass algorithm"):
        pass_tick : float = time()
        direction : str = "f" if j % 2 == 0 else "b"
        pass_range : range = get_pass_range(direction,frames_length)
        prev_img = None
        is_temporal_loss : bool = temporal_loss_after_n_passes >= j
        config["video_mode"] = is_temporal_loss
        for i in pass_range:
            index_d : int = i - 1 if direction == "f" else i + 1
            if direction == "f" and i == 0 or direction == "b" and i - 1 == frames_length:
                prev_img = combination_frames[i]
                stylize_frames[i] = prev_img
                
            else:
                warp_mask = masks[index_d]
                next_img = combination_frames[index_d]
                reverse_flow = True if direction == "b" else False
                warp_img = warp_flow(next_img,flows[index_d],reverse_flow)
                first_mul = blend_weight * warp_mask * warp_img
                ones_res = tf.ones_like(warp_mask)
                neg_prev_mask = tf.subtract(ones_res, warp_mask) 
                second_mul = (neg_blend_weight * ones_res) + (blend_weight * neg_prev_mask) * prev_img
                final_result = tf.add(first_mul,second_mul)
                prev_img = combination_frames[i]
                config["combination_frame"] = final_result
                generated_frames, best_frame, log_data = loop_manager.training_loop(content_path=combination_frames[i],  style_path=style_image,config=config,)
                if not generated_frames or not best_frame or not log_data:
                    print("Error: optimization loop failed. Skipping")
                    continue
                stylize_frames[i] = best_frame.get_image()
        pass_end : float = time()
        duration : float = pass_end - pass_tick
        pass_time.append(duration)
    end : float = time()
    total_pass_duration : float = end - start
    print(f"Multi-pass process ({total_pass_duration :.2f}) seconds")
    return stylize_frames

