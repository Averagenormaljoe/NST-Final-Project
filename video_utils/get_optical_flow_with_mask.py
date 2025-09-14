import traceback
from shared_utils.exception_checks import none_check
from video_utils.helper.get_flow_and_wrap import get_flow_and_wrap, prepare_mask
def get_optical_flow_with_mask(config, combination_image):
    try:
        none_check(config, "config")
        none_check(combination_image, "combination_image")
        config = get_flow_and_wrap(config)
        flow = config.get("flow", None)
        verbose = config.get("verbose", 0)
        video_mode = config.get("video_mode",False)
        if flow is None or not video_mode:
            if verbose >= 1 and video_mode:
                print("ERROR: flow not provided in config")
            return config
        config = prepare_mask(config, combination_image)
    except Exception as e:
        traceback.print_exc()
        mes = f"Error for 'get_optical_flow_with_mask': {e}"
        print(mes)  

    return config