import numpy as np
from video_utils.mask import get_optimal_flow
def process_flow_on_frames(config,combination_image):
       frames = config.get("frames", [])
       if len(frames) > 0:
            prev_frame = frames[-1]
            numpy_combination_image = np.squeeze(combination_image.numpy())
            flow = get_optimal_flow(prev_frame, numpy_combination_image)
            config["flow"] = flow
       return config