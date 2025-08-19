from video_utils.helper.get_flow_and_wrap import get_flow_and_wrap, prepare_mask
from video_utils.mask import warp_previous_frames
def get_flow_and_mask(config, combination_image):
    config = get_flow_and_wrap(config)
    config = warp_previous_frames(config, combination_image=combination_image)
    config = prepare_mask(config, combination_image)
    return config