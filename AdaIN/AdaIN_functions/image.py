from video_utils.video_helper import image_read
import cv2
import numpy as np
import tensorflow as tf
def load_image(path : str):
    style_im = cv2.imread(path)
    style_im = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    style_im = image_read(style_im)
    return style_im

# Code from https://medium.com/@sanansuleyman/style-video-with-neural-style-transfer-d10a35cf0e3
# Website: Medium
# Title: Style Video with Neural Style Transfer
# Author: Sanan Suleymanov
# Date: Jan 28, 2023
def tensor_toimage(tensor : tf.Tensor):
  tensor =tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0]==1
    tensor=tensor[0]
  return tensor
# End of code adaption