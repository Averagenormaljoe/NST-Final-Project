import json
from shared_utils.file_nav import get_base_name


def save_metadata(metadata, file_name):
    with open(file_name, 'w') as f:
        json.dump(metadata, f,sort_keys=True, indent=4)

def prepare_metadata(config, file_paths,extra_dict):
    content_image, style_image = file_paths
    metadata = {
        "content_image": get_base_name(content_image),
        "style_image": get_base_name(style_image),
        "optimizer": config.get('optimizer', 'adam'),
        "preserve_color": config.get('preserve_color', False),
        "noise": config.get('noise', False),
        "clip": config.get('clip', False),
        "lr": config.get('lr', 1.0),
        "size": config.get('size', (400, 400)),
        "content_weight" : config.get("c_weight", 2.5e-8),
        "style_weight" : config.get("s_weight", 1e-6),
        "total_variation_weight" : config.get("tv_weight", 1e-6)
    }
    if extra_dict:
        metadata.update(extra_dict)
    return metadata
