from video_utils.helper.get_flow_and_wrap import get_flow_and_wrap, prepare_mask
from video_utils.mask import warp_previous_frames
def get_flow_and_mask(config, combination_image):
    config = get_flow_and_wrap(config)
    flow = config.get("flow", None)
    verbose = config.get("verbose", 0)
    if flow is None:
        if verbose >= 1:
            print("ERROR: flow not provided in config")
        return config
    config = warp_previous_frames(config,flow)
    config = prepare_mask(config, combination_image)
    return config