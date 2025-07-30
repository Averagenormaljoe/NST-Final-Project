
import os
def video_style_transfer(config):
    """Implements neural style transfer on a video using a style image, applying provided configuration."""
    if config.get('file_dir') is not None:
        file_dir = config.get('file_dir')
        content_video_path = os.path.join(file_dir, config.get('content_filename'))
        style_path = os.path.join(file_dir, config.get('style_filename'))
        output_dir = config.get('output_dir') if config.get('output_dir') is not None else file_dir
    else:
        output_dir = config.get('output_dir')
        content_video_path = config.get('content_filepath')
        style_path = config.get('style_path')
    
    output_size = config.get('output_frame_size')
    if output_size is not None:
        if len(output_size) > 1: 
            output_size = tuple(output_size)
        else:
            output_size = output_size[0]
    
    verbose = not config.get('quiet')

    if not os.path.exists(os.path.join(output_dir, "content_frames")):
        os.makedirs(os.path.join(output_dir, "content_frames"))

    if verbose:
        print("Loading content video...")

    cap = cv2.VideoCapture(content_video_path)
    # retrieve metadata from content video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    content_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print(f"ERROR: could not retrieve frames from content video at path: '{content_video_path}'.")
        return

    # extract frames from content video
    for i in range(total_frames):
        success, img = cap.read()
        if success:
            cv2.imwrite(os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg"), img)
        else:
            print(F'ERROR: {os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")} failed to be extracted.')
            return

    cap.release()

    if verbose:
        print("Frames successfully extracted from content video.")
        print()
        print("Performing image style transfer for each frame...")

    if not os.path.exists(os.path.join(output_dir, "transferred_frames")):
        os.makedirs(os.path.join(output_dir, "transferred_frames"))

    # perform image style transfer with each content frame and style image
    for i in range(total_frames):
        content_frame_path = os.path.join(output_dir, "content_frames", f"frame-{i+1:08d}.jpg")
        output_frame_path = os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg")
        success = _image_style_transfer(content_frame_path, style_path, output_frame_path, output_size)

        if verbose:
            if success:
                print(f'\tImage style transfer success for frame {content_frame_path}.')
            else:
                print(f'\tWarning: Image style transfer failed for frame {content_frame_path}.')
                return
    
    if verbose:
        print("Image style transfer complete.")
        print()
        print("Synthesizing video from transferred frames...")
    
    content_video_name, _ = get_image_name_ext(content_video_path)
    style_img_name, _ = get_image_name_ext(style_path)
    output_video_path = os.path.join(output_dir, f"nst-{content_video_name}-{style_img_name}-final.mp4")

    output_frame_height, output_frame_width, _ = cv2.imread(os.path.join(output_dir, "transferred_frames", "transferred_frame-00000001.jpg")).shape
    output_fps = config.get('fps') if config.get('fps') is not None else content_fps
    # synthesize video using transferred content frames
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, cv2_fourcc, output_fps, (output_frame_width, output_frame_height), True)

    for i in range(total_frames):
        frame = cv2.imread(os.path.join(output_dir, "transferred_frames", f"transferred_frame-{i+1:08d}.jpg"))
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()

    if verbose:
        print(f'Video successfully synthesized to {output_video_path}.')