import os
import cv2
from requests import get
from shared_utils.file_nav import get_base_name
from tqdm import trange
from shared_utils.helper import create_dir
def get_video_paths(config):

    output_dir = config.get('output_dir')
    create_dir(output_dir)
    content_video_path = config.get('content_path')
    style_path = config.get('style_path')
    return content_video_path, style_path, output_dir

def get_video_details(cap):
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    return total_frames, h, w, content_fps

def get_frame_dir():
    return "content_frames", "transferred_frames"

def get_frame_limit(config, total_frames):
    frames_limit = config.get('frames_limit', total_frames)
    if frames_limit > total_frames:
        frames_limit = total_frames
    return frames_limit

def write_frames(config):

    
  
    verbose = config.get('verbose', False)
    frames_dir, transferred_dir = get_frame_dir()
    content_prefix = config.get('content_prefix', 'frame')
    extension = config.get('extension', 'jpg')
    content_video_path, style_path, output_dir = get_video_paths(config)
    content_frames_dir = os.path.join(output_dir, frames_dir)
    create_dir(content_frames_dir)
    if verbose:
        print("Loading content video...")

    cap = cv2.VideoCapture(content_video_path)
    # retrieve metadata from content video
    total_frames, h, w, content_fps = get_video_details(cap)
    frames_limit = get_frame_limit(config, total_frames)
    if total_frames == 0:
        print(f"ERROR: could not retrieve frames from content video at path: '{content_video_path}'.")
        return
    # extract frames from content video
    for i in trange(frames_limit, desc="Extracting frames from content video", disable=not verbose):
        ret, frame = cap.read()
        frame_i = f"{i+1:08d}"
        path = os.path.join(content_frames_dir, f"{content_prefix}-{frame_i}.{extension}")
        if ret:
            cv2.imwrite(path, frame)
        else:
            print(F'ERROR: {path} failed to be extracted.')
            return

    cap.release()
    if verbose:
        print("Frames successfully extracted from content video.")
        print()
        print("Performing image style transfer for each frame...")
    
    return total_frames, h, w, content_fps


def save_output_video(config, video_details):
    verbose = config.get('verbose', False)
    total_frames,h,w, content_fps = video_details
    content_video_path, style_path, output_dir = get_video_paths(config)
    content_video_name = get_base_name(content_video_path)
    style_img_name = get_base_name(style_path)
    output_extension = config.get('output_extension', 'mp4')
    output_video_path = os.path.join(output_dir, f"nst-{content_video_name}-{style_img_name}-final.{output_extension}")
    frames_dir, transferred_dir = get_frame_dir()
    transformed_prefix = config.get('transformed_prefix', 'transferred_frame')
    extension = config.get('extension', 'jpg')
    first_output_frame = os.path.join(output_dir, transferred_dir, f"{transformed_prefix}-{1:08d}.{extension}")
    output_frame_height, output_frame_width, _ = cv2.imread(first_output_frame).shape
    if not os.path.exists(first_output_frame):
        print(f"ERROR: First output frame not found at {first_output_frame}")
        return
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_frame_width, output_frame_height), True)
    frames_limit = get_frame_limit(config, total_frames)
    for i in trange(frames_limit, desc="Combining the stylized frames of the video together", disable=not verbose):
        frame = cv2.imread(os.path.join(output_dir, transferred_dir, f"{transformed_prefix}-{i+1:08d}.{extension}"))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')
        
def normalize_output_size(output_size):
    if output_size is not None:
        resized_output_size = tuple(output_size) if len(output_size) > 1 else output_size[0]
        return resized_output_size
    
        
def video_style_transfer(config,video_details,style_func):
    verbose = config.get('verbose', False)
    output_size = config.get('size')
    output_size = normalize_output_size(output_size)
    config['size'] = output_size
    total_frames,h,w, content_fps = video_details
    content_video_path, style_path, output_dir = get_video_paths(config)
    frames_dir, transferred_dir = get_frame_dir()
    content_frames_dir = os.path.join(output_dir,  frames_dir)
    transferred_frames_dir = os.path.join(output_dir, transferred_dir)
    create_dir(transferred_frames_dir)
    create_dir(content_frames_dir)
    prev_frames = []
    content_prefix = config.get('content_prefix', 'frame')
    transformed_prefix = config.get('transformed_prefix', 'transferred_frame')
    extension = config.get('extension', 'jpg')
    logs = []
    frames_limit = get_frame_limit(config, total_frames)
    # perform image style transfer with each content frame and style image
    for i in trange(frames_limit, desc="Performing style transfer for each frame", disable=not verbose):
        frame_i = f"{i+1:08d}"
        content_frame_path = os.path.join(content_frames_dir, f"{content_prefix}-{frame_i}.{extension}")
        if not os.path.exists(content_frame_path):
            print(f"Missing content frame: {content_frame_path}")
            continue
        output_frame_path = os.path.join(transferred_frames_dir, f"{transformed_prefix}-{frame_i}.{extension}")
        config['output_path'] = output_frame_path
        config["frames"] = prev_frames
        results = style_func(content_frame_path, style_path, config=config)
        

        if verbose:
            if results:
                print(f'\tImage style transfer success for frame {content_frame_path}.')
            else:
                print(f'\tWarning: Image style transfer failed for frame {content_frame_path}.')
                return
        generated_images, best_image,log_data = results
        prev_frames.append(best_image.get_image())
        logs.append(log_data)
    if verbose:
        print("Image style transfer complete.")
        print()
        print("Synthesizing video from transferred frames...")
    config["logs"] = logs
    return video_details
    
    
    
    
def execute_video_style_transfer(config, style_func):
    verbose = config.get('verbose', False)
    if verbose:
        print("Starting video style transfer...")
    
    video_details = write_frames(config)
    if video_details is None:
        print("Failed to write frames. Exiting...")
        return
    
    video_details = video_style_transfer(config, video_details, style_func)
    if video_details is None:
        print("Video style transfer failed. Exiting...")
        return
    
    save_output_video(config, video_details)
    
    if verbose:
        print("Video style transfer completed successfully.")