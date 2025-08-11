import numpy as np
from PIL import Image

def load_image(image_path, dim=None, resize=False):
    img = Image.open(image_path)
    if dim:
        if resize:
            img = img.resize(dim)
        else:
            img.thumbnail(dim)
    img = img.convert("RGB")
    return np.array(img)