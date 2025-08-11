import tensorflow as tf
from functools import reduce
from model_components.ReflectionPadding2D import ReflectionPadding2D
class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs, style_vector):
        batch, rows, cols, channels = [i for i in inputs.get_shape()]
        mu, var = tf.nn.moments(inputs, [1, 2], keepdims=True)
     
        epsilon = 1e-3
        
        
        shift = []
        scale = []
        var_shape = [channels]
        for i in range(len(style_vector)):
            shift.append(tf.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.)))
            scale.append(tf.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.)))
        normalized = (inputs - mu) / tf.sqrt(var + epsilon) **(.5)
         
        idx = [i for i, x in enumerate(style_vector) if not x == 0]
        
        style_scale = reduce(tf.add, [scale[i]*style_vector[i] for i in idx]) / sum(style_vector)
        style_shift = reduce(tf.add, [shift[i]*style_vector[i] for i in idx]) / sum(style_vector)
        output = style_scale * normalized + style_shift
        return output
    
    
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