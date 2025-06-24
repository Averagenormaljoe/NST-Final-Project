import numpy as np
import tensorflow as tf
def preprocess_image(image_path : str,img_height: int, img_width: int):
 img = keras.utils.load_img(
 image_path, target_size=(img_height, img_width))
 img = keras.utils.img_to_array(img)
 img = np.expand_dims(img, axis=0)
 img = keras.applications.vgg19.preprocess_input(img)
 return tf.convert_to_tensor(img, dtype=tf.float32)

def deprocess_image(img,img_height : int, img_width : int):
 img = img.reshape((img_height, img_width, 3))
 img[:, :, 0] += 103.939
 img[:, :, 1] += 116.779
 img[:, :, 2] += 123.68
 img = img[:, :, ::-1]
 img = np.clip(img, 0, 255).astype("uint8")
 return img

def content_loss(base_img, combination_img):
 return tf.reduce_sum(tf.square(combination_img - base_img))


def gram_matrix(x):
 x = tf.transpose(x, (2, 0, 1))
 features = tf.reshape(x, (tf.shape(x)[0], -1))
 gram = tf.matmul(features, tf.transpose(features))
 return gram

def style_loss(style_img, combination_img,img_height: int, img_width : int):
 S = gram_matrix(style_img)
 C = gram_matrix(combination_img)
 channels = 3
 size = img_height * img_width
 return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))