import tensorflow as tf

def add_noise_to_image(image,noise_strength : float =0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_strength, dtype=image.dtype)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 255.0)