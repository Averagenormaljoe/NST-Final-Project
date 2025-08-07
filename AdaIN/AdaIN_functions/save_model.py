import os
from typing import Optional
import tensorflowjs as tfjs
import tensorflow as tf
import zipfile
def save_tensorflowjs_model(model):
    tfjs.converters.save_keras_model(model, "model_tfjs")
    
def save_model(model):
    # save each part of the model in the '.keras' format.
    model.encoder.save("encoder.keras")
    model.decoder.save("decoder.keras")
    model.loss_net.save("loss_net.keras")
    model.save("model.keras")


def save_pb_model(model):
    # model path
    model_dir = "saved_model_pb"
    # save the model in the '.pb' format
    tf.saved_model.save(model, model_dir)
    # save the weights
    model.save_weights('adaptive_weights')
    
def legacy_save_model(model):
    # save each part of the model in the '.h5' format.
    model.encoder.save_weights("encoder.h5")
    model.decoder.save_weights("decoder.h5")
    model.loss_net.save_weights("loss_net.h5")
    model.save_weights("model.h5")
    
def zip_model_files(file_paths : Optional[list[str]] = None):
    if file_paths is None:
        file_paths = ['encoder.keras', 'decoder.keras', 'loss_net.keras', 'model.keras']
    with zipfile.ZipFile('model.zip', 'w') as zipf:
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file)