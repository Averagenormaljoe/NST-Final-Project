def get_optimizer(name: str, learning_rate: float):
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
    elif name == 'sgd':
        return SGD(keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=100, decay_rate=0.96))
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif name == 'adamax':
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif name == 'nadam':
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif name == 'radam':
        return tf.keras.optimizers.RAdam(learning_rate=learning_rate)
    elif name == 'lbfgs':
        return tfp.optimizer.lbfgs_minimize
        
    else:
        raise ValueError(f"Unsupported optimizer: {name}")