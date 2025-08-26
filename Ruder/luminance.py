
import tensorflow as tf
from video_utils.mask import warp_flow
from shared_utils.losses import  apply_mask_and_sum, square_and_sum, square_or_l2
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

def warping_constraint(img):
    xyz = get_xyz(img)
    r,g,b = get_channels(img)
    x,y,z = get_channels(xyz)
    luminance = get_luminance(r,g,b)
    y_with_luminance = y + luminance 
    return x, y_with_luminance, z

def get_channels(img):
    row, col, ch = tf.shape(img)
    if ch == 3:
        r, g, b = tf.unstack(tf.cast(img, tf.float32), axis=-1)
    else:
        r, g, b, a = tf.unstack(tf.cast(img, tf.float32), axis=-1)
    return  r,g,b

def wrap_images_prior_luminance_f1(prev_img,curr_img,prev_stylize_img,curr_stylize_img,mask,flow):
    prev_warped_stylize_img = warp_flow(prev_stylize_img, flow)
    prev_warped_img = warp_flow(prev_img, flow)
    mean = temporal_relative_luminance_f1(prev_warped_img,curr_img, prev_warped_stylize_img,curr_stylize_img,mask,flow)
    return mean


def temporal_relative_luminance_f1(prev_warped_img,curr_img, prev_warped_stylize_img,curr_stylize_img,mask):
    r, g, b = get_channels(curr_stylize_img)
    img_channels = [r,g,b]
    y = get_luminance(r,g,b)
    sum = tf.zeros(shape=())
    for c in img_channels:
        stylize_diff = curr_stylize_img -  prev_warped_stylize_img
        color_stylize_diff = stylize_diff * c
        
        diff = curr_img - prev_warped_img
        luminance_diff = diff * y
        
        final_diff = color_stylize_diff - luminance_diff
    
        sum += square_and_sum(curr_stylize_img, final_diff, mask)
    mean = tf.reduce_mean(sum)
    return  mean




def temporal_relative_luminance_f2(prev_warped_img,curr_img, prev_warped_stylize_img,curr_stylize_img,mask):
    x,Y,z = warping_constraint(curr_stylize_img)
    sum = tf.zeros(shape=())

    stylize_diff = curr_stylize_img -  prev_warped_stylize_img
    color_stylize_diff = stylize_diff * Y
    
    diff = curr_img - prev_warped_img
    luminance_diff = diff * Y
    
    final_diff = color_stylize_diff - luminance_diff
    
    final_diff_l2 += square_or_l2( final_diff,True)
    
    xy_diff =  stylize_diff * x * z
    xy_diff_l2 = square_or_l2(xy_diff,True)
    added_diff = final_diff_l2 + xy_diff_l2
    final_sum = apply_mask_and_sum(added_diff)
    mean = tf.reduce_mean(final_sum)
    return  mean



