from shared_utils.HardwareLogger import HardwareLogger
from shared_utils.compute_custom_losses import CustomLosses
class LoopLogger():
    def __init__(self, config):
        self.hardware_logger = HardwareLogger()
        self.custom_losses = CustomLosses(config.get("lipis_loss_fn", "alex"))
    def end_training(self):
        self.hardware_logger.on_training_end()
        log_data = self.hardware_logger.get_log().copy()
        self.hardware_logger.clear_log()
        return log_data
    def log_save(self, t_loss, i):
        self.hardware_logger.log_loss(t_loss,i)
        self.hardware_logger.log_hardware()
        self.hardware_logger.log_end_check()
    def log_image(self, img, i, content_name: str, style_name: str):
        self.hardware_logger.log_image_paths(content_name, style_name)
    def sum_metrics(self,metrics_list) -> dict:
        dict_sum = {}
        for d in metrics_list:
            for k, v in d.items():
                float_val = float(v.numpy()) if hasattr(v, 'numpy') else float(v)
                dict_sum[k] = dict_sum.get(k, 0) + float_val
        return dict_sum
    def get_custom_metrics(self,base_image, combination_image) -> dict:
        metrics_dict = {}
        includes : list[str] = ["ssim", "psnr", "ms_ssim", "lpips"]
        # compute additional losses if specified
        custom_metrics = self.custom_losses.compute_custom_losses(base_image,combination_image,includes=includes)
        metrics_dict.update(custom_metrics)
        return metrics_dict
    def log_metrics(self, metrics_list, base_image, combination_image,metrics_dict) -> None:
        dict_sum = self.sum_metrics(metrics_list)
        custom_metrics = self.get_custom_metrics(base_image, combination_image)
        dict_sum.update(custom_metrics)
        if metrics_dict:
            dict_sum.update(metrics_dict)
        for key, value in dict_sum.items():
            self.hardware_logger.append(key, value)