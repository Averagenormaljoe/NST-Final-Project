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
    
    def get_loss(self,name,base_image, combination_image,includes,func,loss_dict={}):
        if name in includes:
            loss = func(base_image, combination_image)
            loss_dict[name] = float(loss)
            return loss
        return 0
 
    def compute_custom_losses(self,base_image,combination_image, includes : list[str] = ["ssim", "psnr", "lpips", "ms_ssim"]) ->  dict:
        losses_dict = {}
        if includes is None or len(includes) == 0:
            return losses_dict
        self.get_loss("ssim", base_image, combination_image, includes, ssim_loss, losses_dict)
        self.get_loss("psnr", base_image, combination_image, includes, psnr_loss, losses_dict)
        self.get_loss("ms_ssim", base_image, combination_image, includes, ms_ssim_loss, losses_dict)
        if "lpips" in includes:
            loss_fn = self.get_loss_fn_lpips()
            lpips_loss = get_lpips_loss(base_image, combination_image, loss_fn)
            losses_dict["lpips"] =  float(lpips_loss)
        self.get_loss("artfid", base_image, combination_image, includes, get_artfid_loss, losses_dict)
        fidelity_list = ["fid", "isc", "kid"]
        if any(item in includes for item in fidelity_list):
            fidelity_metrics = get_fidelity_losses(base_image, combination_image, includes)
            losses_dict.update(fidelity_metrics)
        return losses_dict 

def get_fidelity_losses(base_image, combination_image, includes = ["fid", "isc", "kid"]):
    metrics = get_fidelity(base_image, combination_image, includes)
    return metrics