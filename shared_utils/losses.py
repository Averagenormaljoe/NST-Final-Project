import tensorflow as tf

from torch_fidelity import calculate_metrics
import lpips
from pytorch_msssim import ms_ssim

def ssim_loss(x,y,nom_range: int = 1):
    ssim_value = tf.image.ssim(x,y, max_val=nom_range)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(x,y,nom_range: int = 1):
    psnr_value = tf.image.psnr(x,y, max_val=nom_range)
    return 1 - tf.reduce_mean(psnr_value)

def ms_ssim_loss(base_image, combination_image):
    return 1 - ms_ssim(base_image,combination_image, data_range=1.0)  # Assuming normalized images
def get_fid_loss(base_image, combination_image):
    fid_loss = calculate_metrics(
                input1=base_image.numpy(),
                input2=combination_image.numpy(),
                cuda=True,
                verbose=False
            )['frechet_inception_distance']
    return fid_loss
def temporal_loss(base_image, combination_image):
    return base_image - combination_image
    

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