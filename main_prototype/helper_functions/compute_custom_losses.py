from helper_functions.loss_functions import get_isc_loss, get_fid_loss, get_kernel_inception_distance, get_lpips_loss,get_artfid_loss,ssim_loss, psnr_loss
def compute_custom_losses(base_image,combination_image,custom_losses = True, loss_net = "alex", includes : list[str] = ["ssim", "psnr", "lpips"],weights : dict = {}) -> float:
    if includes is None or len(includes) == 0:
        print("The 'includes' list variable is empty, thus no additional losses will be computed.")
        return 0.0
    if custom_losses:    
        loss = 0.0
        if "ssim" in includes:
            ssim_weight = weights.get("ssim", 1.0)
            ssim_loss_value = ssim_loss(combination_image, base_image)
            loss += ssim_loss_value * ssim_weight
        if "psnr" in includes:
            psnr_weight = weights.get("psnr", 1.0)
            psnr_loss_value = psnr_loss(combination_image, base_image)
            loss += psnr_loss_value * psnr_weight 
        if "lpips" in includes:
            lpips_weight = weights.get("lpips", 1.0)
            lpips_loss = get_lpips_loss(base_image, combination_image)
            loss += lpips_loss * lpips_weight
        if "fid" in includes:
            fid_weight = weights.get("fid", 1.0)
            fid_loss = get_fid_loss(base_image, combination_image)
            loss += fid_loss * fid_weight
        if "artfid" in includes:
            artfid_weight = weights.get("artfid", 1.0)
            artfid_loss = get_artfid_loss(base_image, combination_image)
            loss += artfid_loss * artfid_weight
        if "isc" in includes:
            is_weight = weights.get("isc", 1.0)
            is_loss = get_isc_loss(base_image, combination_image)
            loss += is_loss * is_weight
        if "kid" in includes:
            kernel_weight = weights.get("kid", 1.0)
            kernel_loss = get_kernel_inception_distance(base_image, combination_image)
            loss += kernel_loss * kernel_weight
        return loss
    return 0.0