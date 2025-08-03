def save_metadata(metadata, filename):
    with open(filename, 'w') as f:
        json.dump(metadata, f,sort_keys=True, indent=4)

def prepare_metadata(config, image_file_paths):
    metadata = {
        "content_image": get_base_name(image_file_paths[0]),
        "style_image": get_base_name(image_file_paths[1]),
        "optimizer": config.get('optimizer', 'adam'),
        "preserve_color": config.get('preserve_color', False),
        "noise": config.get('noise', False),
        "clip": config.get('clip', False)
    }
    return metadata
