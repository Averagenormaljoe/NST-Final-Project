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

    ]
    for cmd in pip_commands:
        command = cmd.split()
        if "install" in command  and path:
            command.insert(2, f"--target={path}")
        subprocess.run(command)
        
        