import tensorflow as tf
import keras
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


def total_variation_loss(x,img_height: int, img_width : int):
    
    with tf.device("/GPU:0"):
        a = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1, :]
        )
        b = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))