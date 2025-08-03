import json
from shared_utils.file_nav import get_base_name


def save_metadata(metadata, file_name):
    with open(file_name, 'w') as f:
        json.dump(metadata, f,sort_keys=True, indent=4)

def prepare_metadata(config, file_paths):
    metadata = {
        "content_image": get_base_name(file_paths[0]),
        "style_image": get_base_name(file_paths[1]),
        "optimizer": config.get('optimizer', 'adam'),
        "preserve_color": config.get('preserve_color', False),
        "noise": config.get('noise', False),
        "clip": config.get('clip', False)
    }
    return metadata
