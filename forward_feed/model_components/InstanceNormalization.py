
import tensorflow as tf
from keras.saving import register_keras_serializable
@register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        batch, rows, cols, channels = [i for i in inputs.get_shape()]
        mu, var = tf.nn.moments(inputs, [1, 2], keepdims=True)
        shift = tf.Variable(tf.zeros([channels]))
        scale = tf.Variable(tf.ones([channels]))
        epsilon = 1e-3
        normalized = (inputs - mu) / tf.sqrt(var + epsilon)
        return scale * normalized + shift
    
    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)