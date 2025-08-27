import os
import cv2
import keras
from shared_utils.file_nav import get_base_name
from tqdm import trange
from shared_utils.helper import create_dir
from video_utils.mask import multi_pass
from video_utils.video_helper import get_video_details
def get_video_paths(config):

    output_dir = config.get('output_dir')
    create_dir(output_dir)
    content_video_path = config.get('content_path')
    style_path = config.get('style_path')
    return content_video_path, style_path, output_dir



def get_frame_dir():
    return "content_frames", "transferred_frames", "pass_frames"

def get_frame_limit(config, total_frames):
    frames_limit = config.get('frames_limit', total_frames)
    if frames_limit > total_frames:
        frames_limit = total_frames
    return frames_limit

def write_frames(config):

    
  
    verbose = config.get('verbose', False)
    frames_dir, transferred_dir,pass_dir = get_frame_dir()
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
    file_name : str = f"nst-{content_video_name}-{style_img_name}-final.{output_extension}"
    
    frames_dir, transferred_dir,pass_dir = get_frame_dir()
    transformed_prefix = config.get('transformed_prefix', 'transferred_frame')
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps
    is_multi_pass = config.get("is_multi_pass",False)
    frames_limit = get_frame_limit(config, total_frames)
    process_video(file_name,output_dir,transformed_prefix,transferred_dir,frames_limit,output_fps,config)
    if is_multi_pass:
        pass_name : str = f"pass-video-{file_name}"
        pass_prefix = config.get('pass_prefix', 'pass_frame')
        process_video(pass_name,output_dir,pass_prefix,pass_dir,frames_limit,output_fps,config)
        

def process_video(output_file_name,output_dir,prefix,frames_dir,frames_limit,output_fps,config):
    output_video_path = os.path.join(output_dir, output_file_name)
    extension = config.get('extension', 'jpg')
    verbose = config.get('verbose', False)

    first_output_frame = os.path.join(output_dir, frames_dir, f"{prefix}-{1:08d}.{extension}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists(first_output_frame):
        print(f"ERROR: First output frame not found at {first_output_frame}")
        return
    output_frame_height, output_frame_width, _ = cv2.imread(first_output_frame).shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_frame_width, output_frame_height), True)
    for i in trange(frames_limit, desc="Combining the stylized frames of the video together", disable=not verbose):
        frame = cv2.imread(os.path.join(output_dir, frames_dir, f"{prefix}-{i+1:08d}.{extension}"))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')
 
        
def normalize_output_size(output_size):
    if output_size is not None:
        resized_output_size = tuple(output_size) if len(output_size) > 1 else output_size[0]
        return resized_output_size
    return None

def save_frame(output_frame_path,img):
     if output_frame_path is not None and img is not None:
            keras.utils.save_img(output_frame_path, img) 
    
def prepare_transferred_frames(style_func,output_dir,total_frames,style_path,config):
    frames_dir, transferred_dir, pass_dir = get_frame_dir()
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
    
    verbose = config.get('verbose', False)
    for i in trange(frames_limit, desc="Performing style transfer for each frame", disable=not verbose):
        frame_i = f"{i+1:08d}"
        content_frame_path = os.path.join(content_frames_dir, f"{content_prefix}-{frame_i}.{extension}")
        if not os.path.exists(content_frame_path):
            print(f"Missing content frame: {content_frame_path}")
            continue
        output_frame_path = os.path.join(transferred_frames_dir, f"{transformed_prefix}-{frame_i}.{extension}")
        config["frames"] = prev_frames
        results = style_func(content_frame_path, style_path, config=config)
        

        if verbose:
            if results:
                print(f'\tImage style transfer success for frame {content_frame_path}.')
            else:
                print(f'\tWarning: Image style transfer failed for frame {content_frame_path}.')
                return
        generated_images, best_image,log_data = results
        if hasattr(best_image, 'get_image'):
            appended_image = best_image.get_image()
        else:
            appended_image = best_image
        
        prev_frames.append(appended_image)
        save_frame(output_frame_path,appended_image)
        logs.append(log_data)
    if verbose:
        print("Image style transfer complete.")
        print()
        print("Synthesizing video from transferred frames...")
    config["logs"] = logs
      
def apply_multi_pass(config,style_path,pass_frames_dir):
    pass_prefix = config.get('pass_prefix', 'pass_frame')
    verbose = config.get('verbose', False)
    extension = config.get('extension', 'jpg')
    n_pass = config.get("n_pass",3)
    blend_weight = config.get("blend_weight", 0.5) 
    total_flows = config.get("total_flows",[])
    temporal_loss_n = config.get("temporal_loss_n",3)
    multi_pass_frames = multi_pass(n_pass,total_flows,style_path,blend_weight,temporal_loss_n,config=config)
    number_of_pass_frames = len(multi_pass_frames)
    for i in trange(number_of_pass_frames, desc="Performing multi-pass for each frame", disable=not verbose):
        frame_i = f"{i+1:08d}"
        output_frame_path = os.path.join(pass_frames_dir, f"{pass_prefix}-{frame_i}.{extension}")
        next_image =  multi_pass_frames[i]
        save_frame(output_frame_path,next_image)
    if verbose:
        print("Finished multi pass step of the codebase.")
          
      
        
def video_style_transfer(config,video_details,style_func):
    verbose = config.get('verbose', False)
    output_size = config.get('size')
    output_size = normalize_output_size(output_size)
    config['size'] = output_size
    if not video_details:
        if verbose:
            print("ERROR: Not details (total_frames,h,w, fps) has been provided for the video. Exiting video style transfer process.")
        return video_details
    total_frames,h,w, content_fps = video_details
    content_video_path, style_path, output_dir = get_video_paths(config)
    frames_dir, transferred_dir, pass_dir = get_frame_dir()
    pass_frames_dir = os.path.join(output_dir, pass_dir)
    create_dir(pass_frames_dir)
    
    is_multi_pass = config.get("is_multi_pass",False)
    # perform image style transfer with each content frame and style image
    prepare_transferred_frames(style_func,output_dir)

    if is_multi_pass:
        apply_multi_pass(config,style_path,pass_frames_dir,total_frames,style_path,config)

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