import tensorflow as tf

def tf_decode_jpeg(image_path):
    image = tf.io.read_file(image_path)
    channels : int = 3
    image = tf.image.decode_jpeg(image, channels=channels)
    return image