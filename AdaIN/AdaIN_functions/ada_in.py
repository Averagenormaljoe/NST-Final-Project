import tensorflow as tf
from keras import layers
def get_mean_std(x : tf.Tensor, epsilon : float =1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.math.sqrt(tf.maximum(variance, 0.0) + epsilon)
    return mean, standard_deviation

def get_att(content : tf.Tensor, style : tf.Tensor, att=True):
    if att:
        self_content = self_attention(content, tf.shape(content))
        self_style = self_attention(style, tf.shape(style))
    else:
        self_content = content
        self_style = style
    return ada_in(self_style, self_content, att)

def ada_in(style : tf.Tensor, content : tf.Tensor, att : bool =True):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    self_content, self_style = get_att(content, style, att)
    content_mean, content_std = get_mean_std(self_content)
    style_mean, style_std = get_mean_std(self_style)
    cal = (self_content - content_mean) / (content_std + style_mean)
    t = style_std * cal
    return t

def hw_flatten(x : tf.Tensor):
    shape = tf.shape(x)
    return tf.reshape(x, shape=[shape[0], -1, shape[-1]])

def self_attention(x : tf.Tensor,size : tf.Tensor ):

    tensor_shape = x.shape
    channels = tensor_shape[-1]
    floor_C = channels // 2
    f = layers.Conv2D(floor_C, kernel_size=1, padding='same')(x)  # [bs, h, w, c']
    g = layers.Conv2D(floor_C, kernel_size=1, padding='same')(x)  # [bs, h, w, c']
    h = layers.Conv2D(channels, kernel_size=1, padding='same')(x)    # [bs, h, w, c]
    
    f_flat = hw_flatten(f) 
    g_flat = hw_flatten(g)  
    h_flat = hw_flatten(h)  
    
    s = tf.matmul(g_flat, f_flat, transpose_b=True) 
    axis = -1
    beta = tf.nn.softmax(s, axis=axis) 

    o = tf.matmul(beta, h_flat)  # [bs, N, C]
    reshape_o = tf.reshape(o, shape=size)  # [bs, h, w, C]
    return reshape_o
