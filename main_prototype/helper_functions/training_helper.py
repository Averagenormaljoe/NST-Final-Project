from datetime import datetime
import keras
import numpy as np

def result_save(content_name : str,style_name: str,iterations : int, img: np.ndarray,verbose: int = 0):
    now = datetime.now()
    time_format = "%Y%m%d_%H%M%S"
    now = now.strftime(time_format)
    fname = f"images/{content_name}_{style_name}_{now}_combination_image_at_iteration_{iterations}.png"
    keras.utils.save_img(fname, img) 
    if verbose > 0:
        print("Image saved at iteration {}".format(iterations))