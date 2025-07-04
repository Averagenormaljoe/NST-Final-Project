from ctypes import cast
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import np

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