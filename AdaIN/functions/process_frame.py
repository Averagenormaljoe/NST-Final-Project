from AdaIN.AdaIN_functions.image import tensor_toimage
from AdaIN.AdaIN_functions.stylize_image import stylize_image
from video_utils.video_helper import get_cam,prepare_video_writer,release_video_writer, video_end,image_read, resize_frame
import cv2
IMAGE_SIZE = (224, 224)  
def process_frame(frame, model, style_image,image_size= IMAGE_SIZE):
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_frame = image_read(color_frame)
    resized_image_frame = resize_frame(image_frame, image_size)
    resized_style_image = resize_frame(style_image, image_size)
    stylized_frame = stylize_image(model,resized_image_frame,resized_style_image)
    image = tensor_toimage(stylized_frame)
    return image