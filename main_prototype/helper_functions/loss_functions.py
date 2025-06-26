import tensorflow as tf
import keras
from device_helper import get_device
def content_loss(base_img, combination_img):
 return tf.reduce_sum(tf.square(combination_img - base_img))

def mean_style_loss(style_img, combination_img):
    style_mean = tf.reduce_mean(style_img, axis=[1, 2])
    combination_mean = tf.reduce_mean(combination_img, axis=[1, 2])
    return tf.reduce_mean(tf.square(style_mean - combination_mean))



def gram_matrix(x):
 x = tf.transpose(x, (2, 0, 1))
 features = tf.reshape(x, (tf.shape(x)[0], -1))
 gram = tf.matmul(features, tf.transpose(features))
 return gram

def equal_blends(gram_matrices):
    return sum(gram_matrices) / len(gram_matrices)

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



def high_pass_x_y(image, img_height: int = 224, img_width: int = 224,use_image_size: bool = True):
    if use_image_size:
        x = tf.square(
        image[:, : img_height - 1, : img_width - 1, :] - image[:, 1:, : img_width - 1, :]
        )
        y = tf.square(
        image[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :]
        )
    else:
        x = image[:, :, 1:, :] - image[:, :, :-1, :]
        y = image[:, 1:, :, :] - image[:, :-1, :, :]         
    return x, y

def total_variation_loss_l1(x):
        a,b = high_pass_x_y(x)
        return tf.reduce_sum(tf.abs(a)) + tf.reduce_sum(tf.abs(b))

def total_variation_loss_l2(x):
    a,b = high_pass_x_y(x)
    return tf.reduce_sum(tf.pow(a + b, 1.25))
def total_variation_loss(x,use_l2: bool = False):
    if use_l2:
        return total_variation_loss_l2(x)
    else:
        return total_variation_loss_l1(x)
def ssim_loss(y_true, y_pred):

    ssim_value = tf.image.ssim(y_true, y_pred, max_val=255.0)
    return 1 - tf.reduce_mean(ssim_value)

def psnr_loss(y_true, y_pred):
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=255.0)
    return 1 - tf.reduce_mean(psnr_value)