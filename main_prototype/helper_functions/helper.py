import numpy as np
import tensorflow as tf
import keras
def preprocess_image(image_path : str, img_width: int,img_height: int):
    img = keras.utils.load_img(
    image_path, target_size=(img_height, img_width))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

def deprocess_image(img,img_width : int,img_height : int):
    img = img.reshape((img_height, img_width, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img
