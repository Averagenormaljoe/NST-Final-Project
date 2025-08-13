import keras
import tensorflow as tf

def load_keras_model():
    encoder = keras.models.load_model("encoder")
    decoder = keras.models.load_model("decoder")
    loss_net = keras.models.load_model("loss_net")
    model = keras.models.load_model("model")
    return encoder, decoder, loss_net, model

def load_legacy_model():
    encoder = keras.models.load_model("encoder.h5")
    decoder = keras.models.load_model("decoder.h5")
    loss_net = keras.models.load_model("loss_net.h5")
    model = keras.models.load_model("model.h5")
    return encoder, decoder, loss_net, model

def load_pb_model():
    model_dir = "saved_model_pb"
    model = tf.saved_model.load(model_dir)
    return model