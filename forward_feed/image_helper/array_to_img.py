import numpy as np
from PIL import Image
def array_to_img(array):
    array = np.array(array, dtype=np.uint8)
    array = np.squeeze(array)
    if np.ndim(array) > 3:
        assert array.shape[0] == 1
        array = array[0]
        
    return Image.fromarray(array)