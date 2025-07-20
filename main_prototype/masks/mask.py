import os
import numpy as np
from scipy.misc import imread, imresize, imsave

def load_mask(mask_path, shape):
    mask = imread(mask_path, mode="L")
    width, height, _ = shape
    mask = imresize(mask, (width, height), interp='bicubic').astype('float32')

    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    return mask

