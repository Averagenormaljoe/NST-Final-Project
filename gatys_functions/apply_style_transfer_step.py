from helper_functions.device_helper import get_device
from gatys_functions.compute_loss_and_grads import compute_loss_and_grads
from video_utils.get_optical_flow_with_mask import get_optical_flow_with_mask
from video_utils.helper.reset_warp_frames import reset_warp_frames
from shared_utils.exception_checks import none_check
def apply_style_transfer_step(combination_image, base_image, style_image, optimizer, config : dict = {},device_config : dict = {}):
    none_check(combination_image, "combination_image")
    none_check(base_image, "base_image")
    none_check(style_image, "style_images")
    none_check(config, "config")
    none_check(device_config, "device_config")
    video_mode = config.get("video_mode",True)
    if video_mode:
        config = get_optical_flow_with_mask(config, combination_image)
    GPU_in_use = device_config.get("gpu", 0)
    CPU_in_use = device_config.get("cpu", 0)
    with get_device(GPU_in_use, CPU_in_use):
        loss, grads,all_metrics,metrics_dict = compute_loss_and_grads(
            combination_image, base_image, style_image,config=config, device_config=device_config
        )

    config = reset_warp_frames(config)
    optimizer.apply_gradients([(grads, combination_image)])
    return loss, grads,optimizer,all_metrics,metrics_dict