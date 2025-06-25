import tensorflow as tf
import keras
from device_helper import get_device
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


def compute_style_loss_with_consine_similarity(a, b):
    a_flat = tf.reshape(a, [a.shape[0], -1])
    b_flat = tf.reshape(b, [b.shape[0], -1])
    sim = tf.reduce_mean(1 - tf.keras.losses.cosine_similarity(a_flat, b_flat))
    return sim



def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(x,img_height: int, img_width : int):
    
    with get_device():
        a,b = high_pass_x_y(x)
        return tf.reduce_sum(tf.abs(a)) + tf.reduce_sum(tf.abs(b))

def ssim_loss(y_true, y_pred):

    ssim_value = tf.image.ssim(y_true, y_pred, max_val=255.0)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(y_true, y_pred):
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=255.0)
    return 1 - tf.reduce_mean(psnr_value)