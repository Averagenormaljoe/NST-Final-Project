from datetime import datetime
import keras
import numpy as np
import os
from IPython.display import Image, display
def result_save(content_name : str,style_name: str,iterations : int, img: np.ndarray,verbose: int = 0) -> None:
    now = datetime.now()
    time_format : str = "%Y%m%d_%H%M%S"
    now = now.strftime(time_format)
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    data_name = "combination_image_at_iteration"
    file_extension = ".png"
    fname = f"{output_dir}/{content_name}_{style_name}_{now}_{data_name}_{iterations}.{file_extension}"
    keras.utils.save_img(fname, img) 
    
    if verbose > 0:
        print(f"Image saved at iteration {iterations}")
        display(Image(filename=fname))