import os

def retrieve_style_image(images_path: str):
    extensions = ('.jpg', '.jpeg', '.png')
    style_images = []
    for path, subdirs, files in os.walk(images_path):
        for name in files:
            if name.lower().endswith(extensions):
                full_path = os.path.join(path, name)
                style_images.append(full_path)
    return style_images

# Get the image file paths for the style images.
def get_style_images(on_kaggle : bool = True, path : str = ""):
        final_path = f"/kaggle/input/{path}" if on_kaggle else path
        style_images = retrieve_style_image(final_path)
        return style_images
      
            

def split_style_images(on_kaggle : bool,path : str):
    style_images = get_style_images(on_kaggle, path)
    total_style_images = len(style_images)
    train_split = int(0.8 * total_style_images)
    train_style = style_images[: train_split]
    val_split = int(0.9 * total_style_images)
    val_style = style_images[train_split : val_split]
    test_split = int(0.9 * total_style_images)
    test_style = style_images[test_split :]
    return train_style, val_style, test_style