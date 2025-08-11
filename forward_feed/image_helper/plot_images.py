from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def show_image(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title = title
        

def plot_images_grid(images, num_rows=1):
    n = len(images)
    if n > 1:
        num_cols = np.ceil(n / num_rows)
        fig, axes = plt.subplots(ncols=int(num_cols), nrows=int(num_rows))
        axes = axes.flatten()
        fig.set_size_inches((15, 15))
        for i, image in enumerate(images):
            axes[i].imshow(image)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0])