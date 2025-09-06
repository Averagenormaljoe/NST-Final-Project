import tensorflow as tf
from AdaIN.AdaIN_functions.image import tensor_toimage
from AdaIN.AdaIN_functions.stylize_image import stylize_image
from video_utils.video_helper import image_read, resize_frame
import cv2
def process_frame(frame, model, style_image,image_size : tuple[int,int] =(224, 224) ):
    resized_image_frame = resize_frame(frame, image_size)
    expanded_frame = tf.expand_dims(resized_image_frame, axis=0)
    resized_style_image = resize_frame(style_image, image_size)
    stylized_frame = stylize_image(model,expanded_frame,resized_style_image)
    image = tensor_toimage(stylized_frame)
    return image