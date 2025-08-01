def create_log_dir(content_name : str, style_name : str):
    log_folder : str = "logs/tensorboard"
    create_dir(log_folder)
    log_dir = f"{log_folder}/{content_name}_{style_name}"
    return log_dir