import cv2
import tensorflow as tf
from shared_utils.exception_checks import none_check
from shared_utils.losses import temporal_loss
import numpy as np
import keras_hub

def get_simple_mask(img, flow, reverse=False):
    img_shape = img.shape
    mask_of_ones = np.ones(img_shape)
    warped_mask = warp_flow(mask_of_ones, flow, reverse=reverse)
    filter_value = 0.9999
    filtered_mask = (warped_mask > filter_value)
    float_mask = filtered_mask.astype(np.float32)
    if len(img.shape) == 4:
        float_mask = tf.expand_dims(float_mask, axis=0)
        float_mask = tf.expand_dims(float_mask, axis=-1)
    return float_mask        
    


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


def feature_map_temporal_loss(prev_feature_map, curr_feature_map,flow, mask=None):
    optimal_flow = warp_flow(prev_feature_map,flow) 
    tl = temporal_loss(optimal_flow, curr_feature_map, mask=mask)
    return tl

def temporal_warping_error(prev_stylized_frame, curr_stylized_frame, flow, mask=None):
    warped_prev = warp_flow(prev_stylized_frame, flow)
    twe = temporal_loss(warped_prev, curr_stylized_frame, mask=mask)
    return twe
def warp_previous_frames(previous_stylized_frames, flow):
    none_check(previous_stylized_frames, "previous_stylized_frames")
    none_check(flow, "flow")
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
