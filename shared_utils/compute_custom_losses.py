import tensorflow as tf
from shared_utils.loss_functions import content_loss, style_loss
from shared_utils.losses import get_lpips_loss,ssim_loss, psnr_loss, ms_ssim_loss
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
 
    def compute_custom_losses(self,base_image,combination_image, includes : list[str] = ["SSIM", "psnr", "LPIPS", "MS_SSIM"]) ->  dict:
        losses_dict = {}
        if includes is None or len(includes) == 0:
            return losses_dict
        self.get_loss("SSIM", base_image, combination_image, includes, ssim_loss, losses_dict)
        self.get_loss("psnr", base_image, combination_image, includes, psnr_loss, losses_dict)
        self.get_loss("MS_SSIM", base_image, combination_image, includes, ms_ssim_loss, losses_dict)
        if "LPIPS" in includes:
            loss_fn = self.get_loss_fn_lpips()
            lpips_loss = get_lpips_loss(base_image, combination_image, loss_fn)
            losses_dict["LPIPS"] =  float(lpips_loss)
        fidelity_list = ["fid", "isc", "kid"]
        if "content" in includes:
            c = content_loss(base_image, combination_image)
            losses_dict["content"] = float(c)
        if "style" in includes:
            _, w, h = base_image.shape
            s  = style_loss(base_image, combination_image,w,h)
            losses_dict["style"] = float(s)
        if "tv":
            total_variation_loss = tf.reduce_mean(tf.image.total_variation(combination_image))
            losses_dict["tv"] = float(total_variation_loss)
        return losses_dict 
