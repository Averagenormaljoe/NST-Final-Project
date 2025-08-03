from shared_utils import losses
from shared_utils.losses import get_fidelity_losses, get_lpips_loss,get_artfid_loss,ssim_loss, psnr_loss
def compute_custom_losses(base_image,combination_image,custom_losses = True, loss_net = "alex", includes : list[str] = ["ssim", "psnr", "lpips"],weights : dict = {}) -> tuple[float, dict]:
    losses_dict = {}
    if includes is None or len(includes) == 0:
        return 0.0, losses_dict
    if custom_losses:    
        loss = 0.0
        if "ssim" in includes:
            ssim_weight = weights.get("ssim", 1.0)
            ssim_loss_value = ssim_loss(combination_image, base_image)
            loss += ssim_loss_value * ssim_weight
            losses_dict["ssim"] =  float(ssim_loss_value)
        if "psnr" in includes:
            psnr_weight = weights.get("psnr", 1.0)
            psnr_loss_value = psnr_loss(combination_image, base_image)
            loss += psnr_loss_value * psnr_weight
            losses_dict["psnr"] =  float(psnr_loss_value)
        if "lpips" in includes:
            lpips_weight = weights.get("lpips", 1.0)
            lpips_loss = get_lpips_loss(base_image, combination_image)
            loss += lpips_loss * lpips_weight
            losses_dict["lpips"] =  float(lpips_loss)
        fid_losses, fid_metrics = artfid_and_fid_losses(base_image, combination_image, includes, weights)
        loss += fid_losses
        losses_dict.update(fid_metrics)
        isc_losses, isc_metrics = isc_and_kid_losses(base_image, combination_image, includes, weights)
        loss += isc_losses
        losses_dict.update(isc_metrics)
        return loss, losses_dict
    return 0.0, losses_dict 

def fidelity_losses(base_image, combination_image, includes, weights: dict = {}) -> tuple[float, dict]:
    loss = 0.0
    losses_dict = {}
    get_fidelity_losses = get_fidelity_losses(base_image, combination_image, includes)
    if "artfid" in includes:
        artfid_weight = weights.get("artfid", 1.0)
        artfid_loss = output[]
        loss += artfid_loss * artfid_weight
        losses_dict["artfid"] =  float(artfid_loss)
    if "fid" in includes:
        fid_weight = weights.get("fid", 1.0)
        fid_loss = get_fid_loss(base_image, combination_image)
        loss += fid_loss * fid_weight
        losses_dict["fid"] =  float(fid_loss)
    if "isc" in includes:
        isc_weight = weights.get("isc", 1.0)
        isc_loss = get_isc_loss(base_image, combination_image)
        loss += isc_loss * isc_weight
        losses_dict["isc"] =  float(isc_loss)
    if "kid" in includes:
        kid_weight = weights.get("kid", 1.0)
        kid_loss = get_kernel_inception_distance(base_image, combination_image)
        loss += kid_loss * kid_weight
        losses_dict["kid"] =  float(kid_loss)
    return loss, losses_dict