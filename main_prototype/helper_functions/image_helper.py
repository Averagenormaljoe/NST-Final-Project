import tensorflow as tf


def clip_0_1(image, min : float = 0.0, max : float = 1.0):
  return tf.clip_by_value(image, clip_value_min=min, clip_value_max=max)


def add_noise_to_image(image,noise_strength : float =0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_strength, dtype=image.dtype)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 255.0)

def normalization_grads(grads, strength= None) -> list:
    norm = tf.linalg.global_norm(grads)
    n_formula = norm + 1e-8
    scale = strength / n_formula if strength else n_formula 
    norm_grads : list = [g * scale for g in grads]
    return norm_grads