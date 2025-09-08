import os
from video_utils.video import execute_video_style_transfer
from shared_utils.file_nav import get_base_name
def get_default_config(output_dir, video_content_path, video_style_path):
       config = {
            "output_dir": output_dir,
            "content_path": video_content_path,
            "style_path": video_style_path,
            "string_optimizer": "adam",
            "verbose": 0,
            "frames_limit": 5,
            "video_mode" : True
            

        }
       return config

def loop_through_videos(apply_model,style_paths, video_content_path="demo_video/man_at_sea_sliced.mp4",prefix = "model", config= {}):
    total_logs = []
    if not os.path.exists(video_content_path):
        print(f"Video content path does not exist: {video_content_path}. Stopping processing.")
        return total_logs
    for video_style_path in style_paths:
        if not os.path.exists(video_style_path):
            print(f"Style video path does not exist: {video_style_path}. Skipping ...")
            continue
        video_name = get_base_name(video_content_path)
        video_style_name = get_base_name(video_style_path)
        name = f"({video_name})_({video_style_name})"
        output_dir = f"../{prefix}_demo_images_{name}/video_output"
        default_config = get_default_config(output_dir, video_content_path, video_style_path)
        config.update(default_config)
        execute_video_style_transfer(config,apply_model)
        logs = config.get("logs", [])
        if logs:
            total_logs.append(logs)
        else:
            print(f"No metrics logs found for 'loop_through_videos'. Path: {video_style_path}")
    return total_logs

