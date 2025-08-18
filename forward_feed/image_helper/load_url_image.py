from io import BytesIO
import numpy as np
from PIL import Image
import requests

def load_url_image(url, dim=None, resize=False):
    headers = {'User-Agent': 'Mozilla/5.0'}
    img_request = requests.get(url, headers=headers)

    img = Image.open(BytesIO(img_request.content))
    if dim:
        if resize:
            img = img.resize(dim)
        else:
            img.thumbnail(dim)
    img = img.convert("RGB")
    return np.array(img)