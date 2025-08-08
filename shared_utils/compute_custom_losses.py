from shared_utils.loss_func import get_loss_fn
from shared_utils.losses import get_fidelity, get_lpips_loss,get_artfid_loss,ssim_loss, psnr_loss, ms_ssim_loss
import lpips
class CustomLosses:
    def __init__(self, loss_fn="vgg"):
        self.setup_lpips_loss_fn(loss_fn)
    def get_loss_fn_lpips(self):
        return self.loss_fn
    def setup_lpips_loss_fn(self, loss_fn : None | str = None):
        if loss_fn is None:
            self.loss_fn = lpips.LPIPS(net='vgg')
        else:
            self.loss_fn = lpips.LPIPS(net=loss_fn)
    
 
    def compute_custom_losses(self,base_image,combination_image, includes : list[str] = ["ssim", "psnr", "lpips", "ms_ssim"]) ->  dict:
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
            loss_fn = self.get_loss_fn_lpips()
            lpips_loss = get_lpips_loss(base_image, combination_image, loss_fn)
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