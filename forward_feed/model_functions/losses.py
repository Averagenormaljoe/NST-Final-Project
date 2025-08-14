import tensorflow as tf

def gram_matrix(x):
    gram = tf.linalg.einsum("bijc,bijd->bcd", x, x)
    return gram / tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)

def style_loss(placeholder, style, weight):
    assert placeholder.shape == style.shape
    s = gram_matrix(style)
    p = gram_matrix(placeholder)
    return weight * tf.reduce_mean(tf.square(s - p))

def content_loss(placeholder, content, weight):
    assert placeholder.shape == content.shape
    return weight * tf.reduce_mean(tf.square(placeholder - content))