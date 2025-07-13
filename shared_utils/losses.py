import numpy as np
import tensorflow as tf
from torch import T
from torch_fidelity import calculate_metrics
import lpips
from pytorch_msssim import ms_ssim

def ssim_loss(x,y,val_range: float = 1):
    ssim_value = tf.image.ssim(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(x,y,val_range: float = 1):
    psnr_value = tf.image.psnr(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(psnr_value)

def ms_ssim_loss(base_image, combination_image,val_range: float = 1.0):
    return 1 - ms_ssim(base_image,combination_image, data_range=vars)  # Assuming normalized images
def get_fid_loss(base_image, combination_image):
    fid_loss = calculate_metrics(
                input1=base_image.numpy(),
                input2=combination_image.numpy(),
                cuda=True,
                verbose=False
            )['frechet_inception_distance']
    return fid_loss


def get_lpips_loss(base_image, combination_image, loss_net='alex'):
    loss_fn = lpips.LPIPS(net=loss_net) 
    distance = loss_fn(base_image, combination_image)
    return tf.reduce_mean(distance)
def get_artfid_loss(base_image, combination_image):
    fid_loss = get_fid_loss(base_image, combination_image)
    distance = get_lpips_loss(base_image, combination_image)
    artfid_value = (distance + 1) * (fid_loss + 1)
    return artfid_value
def get_isc_loss(base_image, combination_image):
    isc_loss = calculate_metrics(
                input1=base_image.numpy(),
                input2=combination_image.numpy(),
                cuda=True,
                verbose=False
            )['inception_score_mean']
    return 1 - isc_loss
def get_kernel_inception_distance(base_image, combination_image):
    kernel_loss = calculate_metrics(
                input1=base_image.numpy(),
                input2=combination_image.numpy(),
                cuda=True,
                verbose=False
            )['kernel_inception_distance_mean']
    return 1 - kernel_loss


def square_or_l2(x, square: bool = True):
    if square:
        return tf.square(x)
    else:
        return tf.nn.l2_loss(x)





def apply_mask(loss, mask=None):
    if mask is not None:
        loss *= mask
    return loss

def apply_mask_and_sum(img,loss, mask=None):
    mask_loss = apply_mask(loss, mask)
    D = float(img.size)
    mse = (1. / D) *  tf.reduce_sum(mask_loss)
    tl = tf.cast(mse, tf.float32)
    return tl

def temporal_loss(prev_stylized_frame, curr_stylized_frame, mask=None, c = None):
    diff = curr_stylized_frame - prev_stylized_frame
    l2_diff = square_or_l2(diff,True)
    tl = apply_mask_and_sum(prev_stylized_frame, l2_diff, mask)
    return tl

def temporal_loss_l2(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss

