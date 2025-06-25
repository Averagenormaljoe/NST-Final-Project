
import tensorflow as tf
import os
IMAGE_SIZE = (224, 224)
def decode_and_resize(image_path):
    """Decodes and resizes an image from the image file path.

    Args:
        image_path: The image file path.

    Returns:
        A resized image.
    """
    image = tf.io.read_file(image_path)
    channels = 3
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def extract_image_from_voc(element):
    """Extracts image from the PascalVOC dataset.

    Args:
        element: A dictionary of data.

    Returns:
        A resized image.
    """
    image = element["image"]
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image

def retrieve_style_image(images_path: str):
    style_images = os.listdir(images_path)
    style_images = [os.path.join(images_path, path) for path in style_images]
    return style_images

# Get the image file paths for the style images.
def get_style_images(on_kaggle : bool = True):
    if on_kaggle:
        style_images = retrieve_style_image("/kaggle/input/best-artworks-of-all-time/resized/resized")
        return style_images
    else:
        style_images = retrieve_style_image("/content/artwork/resized")
        return style_images
    
    
