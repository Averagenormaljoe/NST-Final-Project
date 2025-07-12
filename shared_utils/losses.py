import cv2
import numpy as np
import tensorflow as tf
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


def temporal_loss(prev_stylized_frame, curr_stylized_frame, mask=None):
    diff = curr_stylized_frame - prev_stylized_frame
    l2_diff = tf.square(diff)
    if mask is not None:
        l2_diff *= mask
    mse = tf.reduce_mean(l2_diff)
    return mse

def lap_loss(base_img, stylized_img):

    base_lap_tf = apply_lap_process(base_img)
    stylized_lap_tf = apply_lap_process(stylized_img)


    loss_fn = tf.keras.losses.MeanSquaredError()
    return loss_fn(base_lap_tf, stylized_lap_tf)

def apply_lap_process(img):
    
    
    numpy_img = np.asarray(img, dtype=np.float64)
    base_lap = cv2.Laplacian(numpy_img, ddepth=cv2.CV_64F, ksize=3)

    base_lap_tf = tf.convert_to_tensor(base_lap, dtype=tf.float32)

    return base_lap_tf 