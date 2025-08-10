def check_if_on_colab():
    on_colab = importlib.util.find_spec("google.colab") is not None
    return on_colab 