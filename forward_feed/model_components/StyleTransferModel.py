from model_components.UpsampleLayer import  UpsampleLayer
from model_components.ConvLayer import ConvLayer
from model_components.ResidualLayer import ResidualLayer


class StyleTransferModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(StyleTransferModel, self).__init__(name="StyleTransferModel", **kwargs)
        self.conv2d_1 = ConvLayer(
            filters=32, kernel_size=(9, 9), strides=1, name="conv2d_1_32"
        )
        self.conv2d_2 = ConvLayer(
            filters=64, kernel_size=(3, 3), strides=2, name="conv2d_2_64"
        )
        self.conv2d_3 = ConvLayer(
            filters=128, kernel_size=(3, 3), strides=2, name="conv2d_3_128"
        )
        self.res_1 = ResidualLayer(filters=128, kernel_size=(3, 3), name="res_1_128")
        self.res_2 = ResidualLayer(filters=128, kernel_size=(3, 3), name="res_2_128")
        self.res_3 = ResidualLayer(filters=128, kernel_size=(3, 3), name="res_3_128")
        self.res_4 = ResidualLayer(filters=128, kernel_size=(3, 3), name="res_4_128")
        self.res_5 = ResidualLayer(filters=128, kernel_size=(3, 3), name="res_5_128")
        self.deconv2d_1 = UpsampleLayer(
            filters=64, kernel_size=(3, 3), name="deconv2d_1_64"
        )
        self.deconv2d_2 = UpsampleLayer(
            filters=32, kernel_size=(3, 3), name="deconv2d_2_32"
        )
        self.deconv2d_3 = ConvLayer(
            filters=3, kernel_size=(9, 9), strides=1, name="deconv2d_3_3"
        )
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.deconv2d_1(x)
        x = self.relu(x)
        x = self.deconv2d_2(x)
        x = self.relu(x)
        x = self.deconv2d_3(x)
        x = (tf.nn.tanh(x) + 1) * (255.0 / 2)
        return x

    def print_shape(self, inputs):
        print(inputs.shape)
        x = self.conv2d_1(inputs)
        print(x.shape)
        x = self.relu(x)
        x = self.conv2d_2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.conv2d_3(x)
        print(x.shape)
        x = self.relu(x)
        x = self.res_1(x)
        print(x.shape)
        x = self.res_2(x)
        print(x.shape)
        x = self.res_3(x)
        print(x.shape)
        x = self.res_4(x)
        print(x.shape)
        x = self.res_5(x)
        print(x.shape)
        x = self.deconv2d_1(x)
        print(x.shape)
        x = self.relu(x)
        x = self.deconv2d_2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.deconv2d_3(x)
        print(x.shape)