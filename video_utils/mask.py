from time import time
import cv2
import tensorflow as tf
from tqdm import trange
from video_utils import wrap
from shared_utils.losses import temporal_loss
import tensorflow_hub as hub
import numpy as np
import urllib
import keras_hub
from shared_utils.save_tmp_img import save_tmp_img
def generate_mask(img,segmenter):
  
    mask = segmenter.predict(img)
    return mask
def start_process(img):
    segmenter = get_segmenter()
    mask = generate_mask(img, segmenter)
    

    
   
def process_mask(mask, img):
    mask = tf.image.resize(mask, (img.shape[0], img.shape[1]))
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.grayscale_to_rgb(mask)
    return mask          
    
def get_segmenter(url =  "deeplab_v3_plus_resnet50_pascalvoc"):
    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        url,  num_classes=2
        )
    return segmenter

def convert_to_numpy(img):
    if hasattr(img, 'numpy'):
        img = img.numpy()
    return img

def get_optimal_flow(prev_img, curr_img, config={}):
    prev_image_path = save_tmp_img(prev_img, "prev_image", return_img=True)
    curr_image_path = save_tmp_img(curr_img, "curr_image", return_img=True)
    flow = load_optical_flow(prev_image_path, curr_image_path, config)
    return flow
    
def load_optical_flow(prev_image_path, curr_image_path, config): 
    prev_img = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
    curr_img = cv2.imread(curr_image_path, cv2.IMREAD_GRAYSCALE)
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
    cast_flow[:,:,0] += np.arange(w)
    cast_flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, cast_flow, None, cv2.INTER_LINEAR)
    return res

def feature_map_temporal_loss(prev_feature_map, curr_feature_map,flow, mask=None):
    optimal_flow = warp_flow(prev_feature_map,flow) 
    tl = temporal_loss(optimal_flow, curr_feature_map, mask=mask)
    return tl

def temporal_warping_error(prev_stylized_frame, curr_stylized_frame, flow, mask=None):
    warped_prev = warp_flow(prev_stylized_frame, flow)
    twe = temporal_loss(warped_prev, curr_stylized_frame, mask=mask)
    return twe

def long_term_temporal_loss(curr_stylized_frame, flow, mask=None,previous_stylized_frames=[]):    
    loss = tf.zeros(shape=())
    for prev_frame in previous_stylized_frames:

        twe = temporal_warping_error(prev_frame, curr_stylized_frame, flow, mask=mask)
        loss += twe
    return loss


def multi_pass(n_pass,frames,flow):
    tick = time()
    pass_time = []
        

    for j in trange(0, n_pass, desc=f"Processing passes in multi pass algorithm"):
        direction = "f" if j % 2 == 0 else "b"
        if direction == "f":
            for i in trange(0, frames, desc=f"Processing frames in pass {j+1} ({direction})"):
                if i == 0:
                    init_img = get_content_noise(frames[i])
                else:
                    wrap_img = warp_flow(init_img, flow)
                    init_img = wrap_img + frames[i]
                
                y *=  init_img
        else:
            if direction == "backward":
                for i in trange(0, frames, desc=f"Processing frames in pass {j+1}"):
                    if i == frames:
                        init_img = get_content_noise(frames[i])
        
    tock = time()
    print(f"Multi-pass process ({tock - tick:.2f}) seconds")
    

def get_deeplab():

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

