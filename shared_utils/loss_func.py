import keras


def get_loss_fn(loss_type="mse"):
    if loss_type == "mse":
        return keras.losses.MeanSquaredError()
    elif loss_type == "mae":
        return keras.losses.MeanAbsoluteError()
    elif loss_type == "huber":
        return keras.losses.Huber()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")