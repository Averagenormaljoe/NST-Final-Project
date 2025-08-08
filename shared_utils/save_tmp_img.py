import os
import shutil
from PIL import Image
import tensorflow as tf
def save_tmp_img(image, folder, prefix="tmp", return_img=False, save_tensor=False):
    dir_path = f"/{prefix}/{folder}"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    img_name = "img_dummy.png"
    img_path = os.path.join(dir_path, img_name)
    if save_tensor:
        save_in_tensor(image, img_path)
    else:
        save_in_numpy(image, img_path)
    if return_img:
        return img_path
    return dir_path

def destroy_tmp_img(path):
    if os.path.exists(path):
        shutil.rmtree(path)
            
def save_in_numpy(image, img_path):
    numpy_image = image.squeeze().astype("uint8")
    im = Image.fromarray(numpy_image)
    im.save(img_path)
    
def save_in_tensor(img_tensor,path):
    img_uint8 = tf.image.convert_image_dtype(img_tensor, dtype=tf.uint8)
    encoded = tf.io.encode_png(img_uint8)
    tf.io.write_file(path, encoded) 
         