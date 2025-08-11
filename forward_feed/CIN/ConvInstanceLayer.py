import tensorflow as tf
from forward_feed.CIN.CIN import ConditionalInstanceNormalization
from model_components.ReflectionPadding2D import ReflectionPadding2D
class ConvInstanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_style, in_ch, out_ch,filters, kernel_size, strides=1, **kwargs):
        super(ConvInstanceLayer, self).__init__(**kwargs)
        self.padding = ReflectionPadding2D([k // 2 for k in kernel_size])
        self.conv2d = tf.keras.layers.Conv2D(in_ch, out_ch,filters, kernel_size, strides)
        self.cin = ConditionalInstanceNormalization(num_style=num_style)

    def call(self, inputs,style_codes):
        x = self.padding(inputs)
        x = self.conv(x)
        x = self.cin(x, style_codes)
        x = self.activation(x)
        return x