import tensorflow as tf
from keras import layers
def get_mean_std(x : tf.Tensor, epsilon : float =1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.math.sqrt(tf.maximum(variance, 0.0) + epsilon)
    return mean, standard_deviation

def get_att(style : tf.Tensor,content : tf.Tensor,layers, att=True):
    if att:
        self_content = self_attention(content,layers)
        self_style = self_attention(style,layers)
    else:
        self_content = content
        self_style = style
    return self_style, self_content

def ada_in(style : tf.Tensor, content : tf.Tensor,layers, att : bool =False):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    self_style,self_content = get_att(style,content,layers, att)
    assert self_content.shape == self_style.shape
    content_mean, content_std = get_mean_std(self_content)
    style_mean, style_std = get_mean_std(self_style)
    t = style_std * (self_content - content_mean) / content_std + style_mean
    return t

def hw_flatten(x : tf.Tensor):
    shape = tf.shape(x)
    return tf.reshape(x, shape=[shape[0], -1, shape[-1]])

def self_attention(x : tf.Tensor,layers,add_tensor = True ):


    f_layer,g_layer,h_layer = layers
    f = f_layer(x)  # [bs, h, w, c']
    g = g_layer(x)  # [bs, h, w, c']
    h = h_layer(x)    # [bs, h, w, c]
    h_shape = tf.shape(h)
    f_flat = hw_flatten(f) 
    g_flat = hw_flatten(g)  
    h_flat = hw_flatten(h)  
    
    s = tf.matmul(g_flat, f_flat, transpose_b=True) 
    axis = -1
    beta = tf.nn.softmax(s, axis=axis) 

    o = tf.matmul(beta, h_flat)  # [bs, N, C]
    reshape_o = tf.reshape(o, shape=h_shape )  # [bs, h, w, C]
    if add_tensor:
        return reshape_o + x
    return reshape_o
