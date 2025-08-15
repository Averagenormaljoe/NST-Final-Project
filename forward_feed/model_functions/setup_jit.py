import tensorflow as tf
def setup_jit():
    try:
        tf.config.optimizer.set_jit(True)
    except Exception as e:
        print(f"Error: {e} ")