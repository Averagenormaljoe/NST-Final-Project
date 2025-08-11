from video_utils.mask import  warp_previous_frames
def get_flow_and_wrap(config):
    flow = config.get("flow", None)
    if flow is not None:
        frames = config.get("frames", [])
        warp_frames = warp_previous_frames(frames, flow)
        config["warp_frames"] = warp_frames
    return config                        