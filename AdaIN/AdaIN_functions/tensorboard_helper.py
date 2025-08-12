import subprocess
def review_tensorboard_results(show_tensorboard: bool = True):
    # display the tensorboard results from the training.
    if show_tensorboard:
        port = 6006
        command = f"%tensorboard --logdir logs/fit --port {port}"
        split_command = command.split()
        subprocess.run(split_command, check=True)