from shared_utils import losses
from shared_utils.losses import get_fidelity, get_lpips_loss,get_artfid_loss,ssim_loss, psnr_loss, ms_ssim_loss
import numpy as np
def compute_custom_losses(base_image,combination_image, loss_net = "alex", includes : list[str] = ["ssim", "psnr", "lpips"]) ->  dict:
    losses_dict = {}
    if includes is None or len(includes) == 0:
        return losses_dict
    if "ssim" in includes:
        ssim_loss_value = ssim_loss(combination_image, base_image)
        losses_dict["ssim"] =  float(ssim_loss_value)
    if "psnr" in includes:
        psnr_loss_value = psnr_loss(combination_image, base_image)
        losses_dict["psnr"] =  float(psnr_loss_value)
    if "ms_ssim" in includes:
        ms_ssim_loss_value = ms_ssim_loss(base_image, combination_image)
        losses_dict["ms_ssim"] = float(ms_ssim_loss_value)
    if "lpips" in includes:
        lpips_loss = get_lpips_loss(base_image, combination_image, loss_net=loss_net)
        losses_dict["lpips"] =  float(lpips_loss)
    if "artfid" in includes:
        artfid_loss = get_artfid_loss(base_image, combination_image)
        losses_dict["artfid"] =  float(artfid_loss)
    fidelity_metrics = get_fidelity_losses(base_image, combination_image, includes)
    losses_dict.update(fidelity_metrics)
    return losses_dict 

def get_fidelity_losses(base_image, combination_image, includes = ["fid", "isc", "kid"]):
    metrics = get_fidelity(base_image, combination_image, includes)
    return metrics