import tensorflow as tf
from keras import layers
def get_mean_std(x : tf.Tensor, epsilon : float =1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.math.sqrt(tf.maximum(variance + epsilon, 0.0))
    return mean, standard_deviation

def get_att(style : tf.Tensor,content : tf.Tensor,layers, att=True):
    if att:
        self_content = self_attention(content,layers)
        self_style = self_attention(style,layers)
    else:
        self_content = content
        self_style = style
    return self_style, self_content
#
def ada_in(style : tf.Tensor, content : tf.Tensor,layers, att : bool =False, epsilon: float = 1e-5):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    self_style,self_content = get_att(style,content,layers, att)
    assert self_content.shape == self_style.shape
    content_mean, content_std = get_mean_std(self_content,epsilon)
    style_mean, style_std = get_mean_std(self_style,epsilon)
    t = style_std * (self_content - content_mean) / (content_std + epsilon) + style_mean
    return t


# Code adapted from https://github.com/JianqiangRen/AAMS/blob/master/net/utils.py
# Website: Github
# Title: AAMS, utils.py
# Author: Jianqiang Ren
# GitHub Profile: JianqiangRen
# Date: Feb 27, 2019
def hw_flatten(x : tf.Tensor):
    shape = tf.shape(x)
    return tf.reshape(x, shape=[shape[0], -1, shape[-1]])
# end of code adaption
# Code adapted from https://github.com/JianqiangRen/AAMS/blob/master/net/aams.py
# Website: Github
# Title: AAMS
# Author: Jianqiang Ren
# GitHub Profile: JianqiangRen
# Date: Feb 27, 2019
def get_attention_scores(f_flat, g_flat):
    axis = -1    
    f_channels = tf.shape(f_flat)[-1] 
    d_k = tf.cast(f_channels, tf.float32)
    scaling = tf.math.sqrt(d_k)
    s = tf.matmul(g_flat, f_flat, transpose_b=True) / scaling
    beta = tf.nn.softmax(s, axis=axis) 
    return beta
    
def self_attention(x : tf.Tensor,layers, use_residual: bool = True ):


    f_layer,g_layer,h_layer = layers
    f = f_layer(x)  # [bs, h, w, c']
    g = g_layer(x)  # [bs, h, w, c']
    h = h_layer(x)    # [bs, h, w, c]
    f_flat = hw_flatten(f) 
    g_flat = hw_flatten(g)  
    h_flat = hw_flatten(h)  
    h_shape = tf.shape(h) 
    beta = get_attention_scores(f_flat, g_flat)
    o = tf.matmul(beta, h_flat)  # [bs, N, C]
    reshape_o = tf.reshape(o, [h_shape[0], h_shape[1], h_shape[2], h_shape[3]])
    
    # Add residual connection
    if use_residual:
        reshape_o = reshape_o + x
    return reshape_o
# end of code adaption