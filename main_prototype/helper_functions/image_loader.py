def get_size(base_image_path: str) -> tuple[int, int]:
    original_width, original_height = keras.utils.load_img(base_image_path).size
    img_height = 400
    img_width = round(original_width * img_height / original_height)
    return img_width, img_height