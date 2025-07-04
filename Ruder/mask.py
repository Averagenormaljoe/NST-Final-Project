import cv2
import tensorflow as tf
import tensorflow_hub as hub
def optimal_flow(prev_img, next_img,prepared_flow , type='farneback'):
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
                
    
    output_path = "dp.flo"
    cv2.optflow.writeOpticalFlow(output_path, flow)
