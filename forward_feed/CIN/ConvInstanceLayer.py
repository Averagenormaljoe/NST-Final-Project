import tensorflow as tf
from forward_feed.CIN.CIN import ConditionalInstanceNormalization
from model_components.ReflectionPadding2D import ReflectionPadding2D
class ConvInstanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_style, filters, kernel_size, strides=1, activation='relu', **kwargs):
        super(ConvInstanceLayer, self).__init__(**kwargs)
        pad = [k // 2 for k in kernel_size]
        kernel_size_tuple = (kernel_size, kernel_size)
        self.padding = ReflectionPadding2D(pad)
        
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size_tuple, strides)
        self.cin = ConditionalInstanceNormalization(num_style=num_style)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs,style_codes):
        x = self.padding(inputs)
        x = self.conv2d(x)
        x = self.cin(x, style_codes)
        x = self.activation(x)
        return x