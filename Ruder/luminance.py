
import tensorflow as tf
from tqdm import trange
from video_utils.mask import warp_flow
from shared_utils.losses import  square_and_sum
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
    r,g,b = 1,2,3
    x,y,z = 1,2,3


    luminance = get_luminance(r,g,b)
    y_with_luminance = y + luminance 
    return x, y_with_luminance, z

def temporal_relative_luminance_f1(prev_img,curr_img,prev_stylize_img,curr_stylize_img,mask,flow):
    prev_warped_stylize_img = warp_flow(prev_stylize_img, flow)
    prev_warped_img = warp_flow(prev_img, flow)
    r,g,b = 1,2,3
    img_channels = [r,g,b]
    y = get_luminance(r,g,b)
    sum = tf.zeros(shape=())
    for c in img_channels:
        stylize_diff = curr_stylize_img -  prev_warped_stylize_img
        color_stylize_diff = stylize_diff * c
        
        diff = curr_img - prev_warped_img
        luminance_diff = diff * y
        
        final_diff = color_stylize_diff - luminance_diff
    
        sum += square_and_sum(prev_img, final_diff, mask)
    mean = tf.reduce_mean(sum)
    return  mean

