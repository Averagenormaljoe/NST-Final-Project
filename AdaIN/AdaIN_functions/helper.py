import tensorflow as tf
from shared_utils.image_utils import tf_decode_jpeg
def decode_and_resize(image_path : str, IMAGE_SIZE : tuple):
    """Decodes and resizes an image from the image file path.

    Args:
        image_path: The image file path.

    Returns:
        A resized image.
    """
    image = tf_decode_jpeg(image_path)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def extract_image_from_voc(element, IMAGE_SIZE : tuple):
    """Extracts image from the PascalVOC dataset.

    Args:
        element: A dictionary of data.

    Returns:
        A resized image.
    """
    image = element["image"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image
