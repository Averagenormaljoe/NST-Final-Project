from helper_functions.device_helper import get_device
from gatys_functions.compute_loss_and_grads import compute_loss_and_grads
from video_utils.helper.get_flow_and_wrap import get_flow_and_wrap
from video_utils.helper.reset_warp_frames import reset_warp_frames
from video_utils.mask import get_simple_mask, warp_previous_frames
def apply_style_transfer_step(combination_image, base_image, style_image, optimizer, config : dict = {},device_config : dict = {}):
    config = get_flow_and_wrap(config)
    config = warp_previous_frames(config, combination_image=combination_image)
    flow = config.get("flow", None)
    if flow is None:
        config["mask"] = get_simple_mask(combination_image, flow, reverse=False)
    GPU_in_use = device_config.get("gpu", 0)
    CPU_in_use = device_config.get("cpu", 0)
    with get_device(GPU_in_use, CPU_in_use):
        loss, grads,all_metrics,metrics_dict = compute_loss_and_grads(
            combination_image, base_image, style_image,config=config, device_config=device_config
        )

    config = reset_warp_frames(config)
    optimizer.apply_gradients([(grads, combination_image)])
    return loss, grads,optimizer,all_metrics