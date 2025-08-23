import tensorflow as tf
from model_components.ConvLayer import ConvLayer
from keras.saving import register_keras_serializable
@register_keras_serializable()
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2d_1 = ConvLayer(filters, kernel_size)
        self.conv2d_2 = ConvLayer(filters, kernel_size)
        self.relu = tf.keras.layers.ReLU()
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        residual = inputs
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.add([x, residual])
        return x
    
    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)