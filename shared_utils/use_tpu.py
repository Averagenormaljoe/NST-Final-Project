import tensorflow as tf
def get_tpu_strategy(verbose=True):
    strategy = None

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        if verbose:
            print("TPU devices: ", tf.config.list_logical_devices('TPU'))

        strategy = tf.distribute.TPUStrategy(resolver)
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
        if verbose:
            print("TPU not found, using GPU.")
        else:
            strategy = tf.distribute.get_strategy()
            if verbose:
                print("GPU not found, using CPU.")

    
    return strategy