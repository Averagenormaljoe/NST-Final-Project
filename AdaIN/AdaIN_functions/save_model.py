import os
from typing import Optional
import tensorflow as tf
import zipfile

def save_model(model,save_path):
    model.encoder.save(f"{save_path}/encoder.keras")
    model.decoder.save(f"{save_path}/decoder.keras")
    model.loss_net.save(f"{save_path}/loss_net.keras")
    model.save(f"{save_path}/model.keras")


def save_pb_model(model,save_path):
    # model path
    model_dir = f"{save_path}/saved_model_pb"
    # save the model in the '.pb' format
    tf.saved_model.save(model, model_dir)
    # save the weights
    model.save_weights(f'{save_path}/adaptive_weights')
    
def legacy_save_model(model):
    # save each part of the model in the '.h5' format.
    model.encoder.save_weights("encoder.h5")
    model.decoder.save_weights("decoder.h5")
    model.loss_net.save_weights("loss_net.h5")
    model.save_weights("model.h5")
    
