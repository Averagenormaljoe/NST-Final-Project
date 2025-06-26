from shared_utils.video import image_read
import cv2
import numpy as np
def load_image(path : str):
    style_im = cv2.imread(path)
    style_im = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    style_im = image_read(style_im)
    return style_im


def tensor_toimage(tensor):
  tensor =tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0]==1
    tensor=tensor[0]
  return tensor