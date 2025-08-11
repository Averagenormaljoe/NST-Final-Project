import tensorflow as tf


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs, style_vector):
        batch, rows, cols, channels = [i for i in inputs.get_shape()]
        mu, var = tf.nn.moments(inputs, [1, 2], keepdims=True)
        shift = style_vector[:, 0:channels]
        scale = style_vector[:, channels:2 * channels]
        epsilon = 1e-3
        
        
        shift = []
        scale = []

      
        normalized = (inputs - mu) / tf.sqrt(var + epsilon) **(.5)
        return scale * normalized + shift