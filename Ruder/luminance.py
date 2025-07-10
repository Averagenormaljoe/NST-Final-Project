import numpy as np
import tensorflow as tf
from skimage.color import rgb2xyz
import tensorflow_io as tfio
def get_xyz(img):
    # convert to uint32
    img_float32 = tf.cast(img, tf.float32) / 255.0

    xyz_float32 = tfio.experimental.color.rgb_to_xyz(img_float32)

    # convert back to uint8
    xyz = tf.cast(xyz_float32 * 255.0, tf.uint8)
    return xyz
def get_luminance(r,g,b):
    luminance = (0.2126*r + 0.7152*g + 0.0722*b)
    return luminance

def wraping_constraint(img):
    xyz = get_xyz(img)
    y = get_luminance(1,5,6)
    constraint = xyz + y

def temporal_luminance(prev_img,curr_img,prev_stylize_img,curr_stylize_img,mask,flow):
    w,h = curr_img.shape
    d = w * h
    
    sum = 0
    
    return  sum /d