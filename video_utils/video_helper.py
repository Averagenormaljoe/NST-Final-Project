import cv2
from typing import Union
import numpy as np
import time
import tensorflow as tf

def get_video_details(cap):
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    return total_frames, h, w, content_fps

ImageType = Union[np.ndarray, tf.Tensor]

# Code from https://medium.com/@sanansuleyman/style-video-with-neural-style-transfer-d10a35cf0e3
# Website: Medium
# Title: Style Video with Neural Style Transfer
# Author: Sanan Suleymanov
# Date: Jan 28, 2023
def image_read(image : ImageType) -> tf.Tensor:
  max_dim=512
  image= tf.convert_to_tensor(image, dtype = tf.float32)
  image= image/255.0
  shape = tf.cast(tf.shape(image)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim/long_dim
  new_shape = tf.cast(shape*scale, tf.int32)
  new_image = tf.image.resize(image, new_shape)
  new_image = new_image[tf.newaxis, :]
  
  return new_image
# End of code adaption
def get_cam(video_path: str,camera_mode = True):
    if camera_mode == True:
        cam = cv2.VideoCapture(0) 
    else:
        cam = cv2.VideoCapture(video_path)
    _,frame_height,frame_width, fps = get_video_details(cam)
    return cam, frame_width, frame_height, fps

def prepare_video_writer(output_path: str, frame_width: int, frame_height: int, fps: int,file_format : str = 'mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return out

def release_video_writer(cam,out):
    if out is not None:
        cam.release()
        out.release()
        cv2.destroyAllWindows()
        
def video_end(start_time: float):
    end_time = time.time()
    processing_duration = end_time - start_time
    print(f"Processing completed in {processing_duration:.2f} seconds.")
def resize_frame(frame, target_size):
    if frame.shape[:2] != target_size:
        resized_frame = tf.image.resize(frame, target_size)
        return resized_frame
    return frame