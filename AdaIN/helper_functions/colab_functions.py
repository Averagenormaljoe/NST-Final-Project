def download_libraries():
    !pip install torch_fidelity
    !pip install pytorch_msssim
    !pip install lpips
    !pip install tf2onnx
    !pip install coremltools
    !pip install tensorflowjs
    !pip uninstall tensorflow -y
    !pip install tensorflow[and-cuda]
    !pip install --upgrade tensorflow_decision_forests
