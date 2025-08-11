import tensorflow as tf
from functools import reduce

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
        normalized = (inputs - mu) / tf.sqrt(var + epsilon)
         
        idx = [i for i, x in enumerate(style_vector) if not x == 0]
        
        style_scale = reduce(tf.add, [scale[i]*style_vector[i] for i in idx]) / sum(style_vector)
        style_shift = reduce(tf.add, [shift[i]*style_vector[i] for i in idx]) / sum(style_vector)
        output = style_scale * normalized + style_shift
        return output
    
    
