import traceback
from shared_utils.compute_custom_losses import CustomLosses
def get_custom_metrics(base_image, combination_image) -> dict:
    try:
        metrics_dict = {}
        includes : list[str] = ["SSIM", "psnr", "MS_SSIM", "LPIPS"]
        # compute additional losses if specified
        custom_losses = CustomLosses()
        custom_metrics = custom_losses.compute_custom_losses(base_image,combination_image,includes=includes)
        metrics_dict.update(custom_metrics)
        return metrics_dict
    except Exception as e:
        traceback.print_exc()
        mes = f"Error for 'get_custom_metrics': {e}"
        print(mes)  