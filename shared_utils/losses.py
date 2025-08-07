import shutil
import numpy as np
import tensorflow as tf
from AdaIN.AdaIN_functions import convert
from torch_fidelity import calculate_metrics
import lpips
from PIL import Image
import os
from torchvision.io import read_image

def save_tmp_img(image, folder, prefix="tmp", return_img=False):
    dir_path = f"/{prefix}/{folder}"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    img_name = "img_dummy.png"
    img_path = os.path.join(dir_path, img_name)
    im = Image.fromarray(image)
    im.save(img_path)
    if return_img:
        return img_path
    return dir_path

def ssim_loss(x,y,val_range: float = 1):
    ssim_value = tf.image.ssim(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(x,y,val_range: float = 1):
    psnr_value = tf.image.psnr(x,y, max_val=val_range)
    return 1 - tf.reduce_mean(psnr_value)

def ms_ssim_loss(base_image, combination_image,val_range: float = 1.0):
    mm_ssim = tf.image.ssim_multiscale(base_image,combination_image, max_val=val_range)  
    return 1 - tf.reduce_mean(mm_ssim)

def get_lpips_loss(base_image, combination_image, loss_net='vgg'):
    convert_base_image = base_image if isinstance(base_image, np.ndarray) else base_image.numpy()
    convert_combination_image = combination_image if isinstance(combination_image, np.ndarray) else combination_image.numpy()
    base_img_path = save_tmp_img(convert_base_image, "base",return_img=True)
    combination_img_path = save_tmp_img(convert_combination_image, "combination", return_img=True)
    loss_fn = lpips.LPIPS(net=loss_net) 
    base_image_pt = read_image(base_img_path)
    combination_image_pt = read_image(combination_img_path)
    distance_pt = loss_fn(base_image_pt, combination_image_pt)
    return distance_pt.item()

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
        loss *= mask
    return loss

def apply_mask_and_sum(img,loss, mask=None):
    mask_loss = apply_mask(loss, mask)
    D = float(img.size)
    mse = (1. / D) *  tf.reduce_sum(mask_loss)
    tl = tf.cast(mse, tf.float32)
    return tl

def temporal_loss(prev_stylized_frame, curr_stylized_frame, mask=None):
    diff = curr_stylized_frame - prev_stylized_frame
    l2_diff = square_or_l2(diff,True)
    tl = apply_mask_and_sum(prev_stylized_frame, l2_diff, mask)
    return tl

def square_and_sum(img,diff, mask=None):
    l2_diff = square_or_l2(diff,True)
    tl = apply_mask_and_sum(img, l2_diff, mask)
    return tl

def temporal_loss_l2(x, w, c):
  c = c[np.newaxis,:,:,:]
  D = float(x.size)
  loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
  loss = tf.cast(loss, tf.float32)
  return loss

