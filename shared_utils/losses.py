import numpy as np
import tensorflow as tf
from torch_fidelity import calculate_metrics
import lpips
from torchvision.io import read_image
from shared_utils.save_tmp_img import save_tmp_img  


def ssim_loss(x,y,val_range: float = 1):
    ssim_value = tf.image.ssim(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(x,y,val_range: float = 1):
    psnr_value = tf.image.psnr(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(psnr_value)

def ms_ssim_loss(base_image, combination_image,val_range: float = 1.0):
    mm_ssim = tf.image.ssim_multiscale(base_image,combination_image, max_val=val_range)  
    return 1 - tf.reduce_mean(mm_ssim)

def calculate_lpips_loss(base_image_pt, combination_image_pt, loss_fn=None):
    if loss_fn is None:
        loss_fn = lpips.LPIPS(net='vgg')
    distance_pt = loss_fn(base_image_pt, combination_image_pt)
    return distance_pt.item()

def lpips_pt_convert(image,name):
    numpy_image = image if isinstance(image, np.ndarray) else image.numpy()
    tmp_img_path = save_tmp_img(numpy_image, name,return_img=True)
    image_pt = read_image(tmp_img_path)
    return image_pt


def get_lpips_loss(base_image, combination_image, loss_fn = None):
    base_image_pt = lpips_pt_convert(base_image, "base")
    combination_image_pt = lpips_pt_convert(combination_image, "combination")
    lpips = calculate_lpips_loss(base_image_pt, combination_image_pt, loss_fn)
    return lpips    

def get_fidelity(base_image, combination_image,includes = []) -> dict:
    if includes is None:
        return {}
    metrics = {
        "fid": "fid" in includes,
        "isc": "isc" in includes,
        "kid": "kid" in includes,
    }
    if not any(metrics.values()):
        return {}
    base_tmp = save_tmp_img(base_image.numpy(), "base")
    combination_tmp = save_tmp_img(combination_image.numpy(), "combination")
    results = calculate_metrics(
                input1=base_tmp,
                input2=combination_tmp,
                cuda=True,
                kid_subset_size=1,
                fid=metrics["fid"],
                isc=metrics["isc"],
                 kid=metrics["kid"])
    output = {}
    collect_result = {"fid" : "frechet_inception_distance",
                      "isc" : "inception_score_mean",
                      "kid" : "kernel_inception_distance_mean"}
    for k, v in collect_result.items():
        if metrics[k]:
            output[k] = results[v]
    return output

def get_artfid_loss(base_image, combination_image):
    includes = ["fid"]
    fid_loss = get_fidelity(base_image, combination_image,includes)["fid"]
    distance = get_lpips_loss(base_image, combination_image)
    artfid_value = (distance + 1) * (fid_loss + 1)
    return artfid_value


def square_or_l2(x, square: bool = True):
    if square:
        return tf.square(x)
    else:
        return tf.nn.l2_loss(x)


def apply_mask(loss, mask=None):
    if mask is not None:
        float_loss = tf.cast(loss, tf.float32)
        float_mask = tf.cast(mask, tf.float32)
        loss = tf.multiply(float_loss, float_mask)
    return loss

def apply_mask_and_sum(img,loss, mask=None):
    mask_loss = apply_mask(loss, mask)
    D = tf.size(img, out_type=tf.float32)
    mse = (1. / D) *  tf.reduce_sum(mask_loss)
    tl = tf.cast(mse, tf.float32)
    return tl

def temporal_loss(prev_stylized_frame, curr_stylized_frame, mask=None):
    float32_prev_stylized_frame = tf.cast(prev_stylized_frame, tf.float32)
    float32_curr_stylized_frame = tf.cast(curr_stylized_frame, tf.float32)

    diff = tf.subtract(float32_curr_stylized_frame, float32_prev_stylized_frame)
    l2_diff = square_or_l2(diff,True)
    tl = apply_mask_and_sum(float32_prev_stylized_frame, l2_diff, mask)
    return tl

def square_and_sum(img,diff, mask=None):
    l2_diff = square_or_l2(diff,True)
    tl = apply_mask_and_sum(img, l2_diff, mask)
    return tl

def temporal_loss_v2(x, w, c):
  expand_c = tf.expand_dims(c, axis=0)
  D = tf.size(x, out_type=tf.float32)
  loss = (1. / D) * tf.reduce_sum(expand_c * tf.nn.l2_loss(x - w))
  float_loss = tf.cast(loss, tf.float32)
  return float_loss

