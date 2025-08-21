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
    
    

class CIN(nn.Module):
    """Conditional Instance Norm."""

    def __init__(self, num_style, ch):
        """Init with number of style and channel."""
        super(CIN, self).__init__()
        self.normalize = nn.InstanceNorm2d(ch, affine=False)
        self.offset = tf.Variable(0.01 * tf.random.normal(shape=(1, num_style, ch)), trainable=True))
        
        self.scale = tf.Variable(1.0 + 0.01 * tf.random.normal(shape=(1, num_style, ch)), trainable=True)

    def forward(self, x, style_codes):
        """Forward func."""
        b, c, h, w = x.size()

        x = self.normalize(x)

        gamma = torch.sum(self.scale * style_codes, dim=1).view(b, c, 1, 1)
        beta = torch.sum(self.offset * style_codes, dim=1).view(b, c, 1, 1)

        x = x * gamma + beta

        return x.view(b, c, h, w)
