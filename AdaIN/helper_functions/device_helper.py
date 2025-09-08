import tensorflow as tf
def get_device(GPU_in_use: int = 0, CPU_in_use: int = 0):
    gpus = tf.config.list_physical_devices('GPU')
    tpus = tf.config.list_physical_devices('TPU')
    if tpus:
        device_name : str = f'/TPU:0'
    elif gpus:
        device_name : str = f'/GPU:{GPU_in_use}'
    else:
        device_name : str = f'/CPU:{CPU_in_use}'
        
    return tf.device(device_name)