import re
import tf2onnx
import onnx
import coremltools
import tensorflowjs as tfjs
import tensorflow as tf
import os
from helper import create_dir
def convert_to_onnx(model,image_size=(224, 224),name="StyleMotion"):
    signature =  (tf.TensorSpec((None, *image_size, 3), tf.float32, name="input"),)
    folder_path : str = "onnx"
    create_dir(folder_path)
    output_path : str = f"{folder_path}/{name}.onnx"
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=signature, output_path=output_path)
    onnx.save(onnx_model, output_path)
    print(f"Model saved to {output_path}")
    return onnx_model
def convert_to_core(model,model_name: str = "StyleMotion"):
    core_model = coremltools.converters.keras.convert(model,
        input_names="image",
        image_input_names="image",
        image_scale=1/255.0)
    folder_path : str = "coreml"
    create_dir(folder_path)
    output_path : str = f"{folder_path}/{model_name}.mlmodel"
    core_model.save(output_path)
    return core_model
def save_tensorflowjs_model(model,folder_path : str = "tfjs", name: str = "model_tfjs"):
    output_path : str = f"{folder_path}/{name}"
    create_dir(folder_path)
    tfjs.converters.save_keras_model(model, output_path)
    