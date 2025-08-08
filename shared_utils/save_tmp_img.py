import os
import shutil
from PIL import Image
def save_tmp_img(image, folder, prefix="tmp", return_img=False):
    dir_path = f"/{prefix}/{folder}"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    img_name = "img_dummy.png"
    img_path = os.path.join(dir_path, img_name)
    numpy_image = image.squeeze().astype("uint8")
    im = Image.fromarray(numpy_image)
    im.save(img_path)
    if return_img:
        return img_path
    return dir_path

def destroy_tmp_img(path):
    if os.path.exists(path):
        shutil.rmtree(path)