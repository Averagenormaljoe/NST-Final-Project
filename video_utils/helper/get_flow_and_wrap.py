from video_utils.mask import  get_simple_mask, warp_previous_frames
def get_flow_and_wrap(config):
    flow = config.get("flow", None)
    if flow is not None:
        frames = config.get("frames", [])
        warp_frames = warp_previous_frames(frames, flow)
        config["warp_frames"] = warp_frames
    return config                        
def prepare_mask(config, combination_image):
    flow = config.get("flow", None)
    if flow is not None:
        mask = get_simple_mask(combination_image, flow, reverse=False)
        config["mask"] = mask
        total_masks = config.get("total_masks",[])
        total_masks.append(mask)
        config["total_masks"] = total_masks
    return config