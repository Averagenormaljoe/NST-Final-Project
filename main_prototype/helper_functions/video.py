
import os

import cv2
from requests import get

def get_video_paths(config):
    
    if config.get('file_dir') is not None:
        file_dir = config.get('file_dir')
        content_video_path = os.path.join(file_dir, config.get('content_filename'))
        output_dir = config.get('output_dir') if config.get('output_dir') is not None else file_dir
    else:
        output_dir = config.get('output_dir')
        content_video_path = config.get('content_filepath')
    style_path = config.get('style_filepath')
    return content_video_path, style_path, output_dir

def get_video_details(cap):
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    return total_frames, h, w, content_fps

def write_frames(config):

    
  
    verbose = config.get('verbose', False)
    frames_dir =  "content_frames"
    content_video_path, style_path, output_dir = get_video_paths(config)
    if not os.path.exists(os.path.join(output_dir,  frames_dir)):
        os.makedirs(os.path.join(output_dir,  frames_dir))
    if verbose:
        print("Loading content video...")

    cap = cv2.VideoCapture(content_video_path)
    # retrieve metadata from content video
    total_frames, h, w, content_fps = get_video_details(cap)
    
    if total_frames == 0:
        print(f"ERROR: could not retrieve frames from content video at path: '{content_video_path}'.")
        return

    # extract frames from content video
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, frames_dir, f"frame-{i+1:08d}.jpg"), frame)
        else:
            print(F'ERROR: {os.path.join(output_dir, frames_dir, f"frame-{i+1:08d}.jpg")} failed to be extracted.')
            return

    cap.release()
    if verbose:
        print("Frames successfully extracted from content video.")
        print()
        print("Performing image style transfer for each frame...")
    
    return total_frames, h, w, content_fps

def get_base_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def save_output_video(config, video_details):
    verbose = config.get('verbose', False)
    total_frames,h,w, content_fps = video_details
    content_video_path, style_path, output_dir = get_video_paths(config)
    content_video_name = get_base_name(content_video_path)
    style_img_name = get_base_name(style_path)
    output_video_path = os.path.join(output_dir, f"nst-{content_video_name}-{style_img_name}-final.mp4")

    output_frame_height, output_frame_width, _ = cv2.imread(os.path.join(output_dir, "transferred_frames", "transferred_frame-00000001.jpg")).shape
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, cv2_fourcc, output_fps, (output_frame_width, output_frame_height), True)

    for i in range(total_frames):
        frame = cv2.imread(os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg"))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')
        
        
        
def video_style_transfer(config,video_details,loop_manager):
    verbose = config.get('verbose', False)
    output_size = config.get('size')
    if output_size is not None:
        if len(output_size) > 1: 
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]
    config['size'] = output_size
    total_frames,h,w, content_fps = video_details
    content_video_path, style_path, output_dir = get_video_paths(config)
    if not os.path.exists(os.path.join(output_dir, "transferred_frames")):
        os.makedirs(os.path.join(output_dir, "transferred_frames"))

    # perform image style transfer with each content frame and style image
    for i in range(total_frames):
        content_frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
        output_frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
        config['output_path'] = output_frame_path
        results = loop_manager.training_loop(content_frame_path, style_path, config=config)
    

        if verbose:
            if results:
                print(f'\tImage style transfer success for frame {content_frame_path}.')
            else:
                print(f'\tWarning: Image style transfer failed for frame {content_frame_path}.')
                return
    
    if verbose:
        print("Image style transfer complete.")
        print()
        print("Synthesizing video from transferred frames...")
    return video_details
    
