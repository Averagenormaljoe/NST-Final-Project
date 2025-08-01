import tensorflow as tf

def find_all_gpus():
    # list all physical GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print("CUDA Available:", gpus)
    print("Num GPUs Available: ", len(gpus))
       
def show_cpu():
    cpu_devices = tf.config.list_physical_devices('CPU')
    print("Available CPUs:", cpu_devices)
    
def show_gpu(GPU_in_use):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(gpus[GPU_in_use].name)
    else:
        print("No GPU found")