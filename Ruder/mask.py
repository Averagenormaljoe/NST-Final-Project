from ctypes import cast
import cv2
from requests import get
import tensorflow as tf
from shared_utils.losses import temporal_loss
import tensorflow_hub as hub
import np
import urllib
import keras_hub
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

def optimal_flow(prev_img, next_img,prepared_flow , type='farneback', save_flow=False):
    if type == 'deepflow':
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(prev_img, next_img, None)
    elif type == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif type == 'lucas_kanade':
        flow = cv2.calcOpticalFlowPyrLK(prev_img, next_img, None)
    elif type == 'dis':
        dis = cv2.optflow.createOptFlow_DIS()
        flow = dis.calc(prev_img,next_img, None)

    
                
    if save_flow:
        output_path = "dp.flo"
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


def get_deeplab():

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

