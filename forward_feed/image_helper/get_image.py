from image_helper.load_url_image import get_url_image
from image_helper.load_image import load_image

def get_image(url,input_shape):
    if url.startswith("http"):
        image = get_url_image(url,input_shape)
    else:
        base_image = load_image(url, dim=(input_shape[0], input_shape[1]), resize=True)
        image = base_image / 255.0
    return image