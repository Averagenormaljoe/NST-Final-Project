import tensorflow as tf
from keras.saving import register_keras_serializable
@register_keras_serializable()
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(
            input_tensor,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
            "REFLECT",
        )
    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({
            "padding": self.padding,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)