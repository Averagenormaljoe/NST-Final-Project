import tensorflow as tf
from tensorflow.keras import layers
def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.math.sqrt(tf.maximum(variance, 0.0) + epsilon)
    return mean, standard_deviation


def ada_in(style, content):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def self_attention(x,size ):

    tensor_shape = tf.shape(x)
    channels = tensor_shape[-1]
    floor_C = channels // 2
    f = layers.Conv2D(floor_C, kernel_size=1, padding='same')(x)  # [bs, h, w, c']
    g = layers.Conv2D(floor_C, kernel_size=1, padding='same')(x)  # [bs, h, w, c']
    h = layers.Conv2D(channels, kernel_size=1, padding='same')(x)    # [bs, h, w, c]
    
    f_flat = hw_flatten(f) 
    g_flat = hw_flatten(g)  
    h_flat = hw_flatten(h)  
    
    
    s = tf.matmul(g_flat, f_flat, transpose_b=True)  
    
    beta = tf.nn.softmax(s) 

    o = tf.matmul(beta, h_flat)  # [bs, N, C]
    o = tf.reshape(o, shape=size)  # [bs, h, w, C]
    return o
