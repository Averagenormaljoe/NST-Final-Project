import cv2
from typing import Union
import numpy as np
import tensorflow as tf
def get_cam_details(cam):
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    return frame_width, frame_height, fps

def load_the_video(video_path : str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames



ImageType = Union[np.ndarray, tf.Tensor]

def frame_to_image(image : ImageType) -> tf.Tensor:
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