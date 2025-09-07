import numpy as np
import tensorflow as tf
from video_utils.mask import get_optimal_flow
def process_flow_on_frames(config,combination_image):
       frames = config.get("frames", [])
       video_mode = config.get("video_mode",False)
       if len(frames) > 0 and video_mode:
            prev_frame = frames[-1]
            if hasattr(combination_image, 'numpy'):
               numpy_combination_image = np.squeeze(combination_image.numpy())
            else:
               numpy_combination_image = combination_image
            flow = get_optimal_flow(prev_frame, numpy_combination_image)
            config["flow"] = flow
            total_flows = config.get("total_flows",[])
            total_flows.append(flow)
            config["total_flows"] = total_flows
       return config