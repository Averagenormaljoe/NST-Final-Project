import subprocess
def download_libraries(path : str = "/content/drive/MyDrive/Library"):
    pip_commands = [
        "pip install torch_fidelity",
        "pip install pytorch_msssim",
        "pip install lpips",
        "pip install tf2onnx",
        "pip install coremltools",
        "pip install tensorflowjs",
        "pip uninstall tensorflow -y",
        "pip install tensorflow[and-cuda]",
        "pip install --upgrade tensorflow_decision_forests"
    ]
    for cmd in pip_commands:
        command = cmd.split()
        if path:
            command.insert(2, f"--target=${path}")
        subprocess.run(command)
        
        