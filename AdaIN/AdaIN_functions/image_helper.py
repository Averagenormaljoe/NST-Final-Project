import os
def retrieve_style_image(images_path: str):
    style_images = os.listdir(images_path)
    style_images = [os.path.join(images_path, path) for path in style_images]
    return style_images

# Get the image file paths for the style images.
def get_style_images(on_kaggle : bool = True):
    if on_kaggle:
        style_images = retrieve_style_image("/kaggle/input/best-artworks-of-all-time/resized/resized")
        return style_images
    else:
        style_images = retrieve_style_image("/content/artwork/resized")
        return style_images

def split_style_images(on_kaggle : bool):
    style_images = get_style_images(on_kaggle)
    total_style_images = len(style_images)
    train_style = style_images[: int(0.8 * total_style_images)]
    val_style = style_images[int(0.8 * total_style_images) : int(0.9 * total_style_images)]
    test_style = style_images[int(0.9 * total_style_images) :]
    return train_style, val_style, test_style