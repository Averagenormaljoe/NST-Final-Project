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


def mask_content(content, generated, mask):
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated