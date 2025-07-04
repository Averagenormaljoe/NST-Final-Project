import keras
def get_loss_fn(loss_type="mse"):
    if loss_type == "mse":
        return keras.losses.MeanSquaredError()
    elif loss_type == "mae":
        return keras.losses.MeanAbsoluteError()
    elif loss_type == "huber":
        return keras.losses.Huber()
    elif loss_type == "cosine":
        return keras.losses.CosineSimilarity()
    else:
        print(f"Unknown loss function {loss_type}. Using mse by default.")
        return get_loss_fn("mse")