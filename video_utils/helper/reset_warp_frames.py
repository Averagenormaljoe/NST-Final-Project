from shared_utils.exception_checks import none_check

def reset_warp_frames(config):
    none_check(config, "config")
    config["warp_frames"] = []
    return config