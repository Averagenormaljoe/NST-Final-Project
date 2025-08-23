import tensorflow as tf
from model_components.InstanceNormalization import InstanceNormalization
from model_components.ReflectionPadding2D import ReflectionPadding2D
from keras.saving import register_keras_serializable
@register_keras_serializable()
class UpsampleLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, upsample=2, **kwargs):
        super(UpsampleLayer, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_size = upsample
        
        self.upsample = tf.keras.layers.UpSampling2D(size=upsample)
        self.padding = ReflectionPadding2D([k // 2 for k in kernel_size])
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides)
        self.bn = InstanceNormalization()

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.padding(x)
        x = self.conv2d(x)
        return self.bn(x)
    
    def get_config(self):
        config = super(UpsampleLayer, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "upsample": self.upsample_size
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)