from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

from shared_utils.file_nav import get_base_name
def plot_images(axis, style_image, content_image, nst_image):
    ax_style, ax_content, ax_reconstructed = axis
    ax_style.imshow(style_image)
    ax_style.set_title("Style Image")
    ax_content.imshow(content_image)
    ax_content.set_title("Content Image")
    ax_reconstructed.imshow(nst_image)
    ax_reconstructed.set_title("NST Image")
def inference_style_transfer(model, dataset,stylize_func, num_samples=10,output_dir="stylized_images",batch = 1):
    os.makedirs(output_dir, exist_ok=True)
    sample_index = 0
    metrics_list = []
    sample_dataset = dataset if batch == 0 else dataset.take(batch)
    for style, content in sample_dataset:
        reconstructed_image, metrics = stylize_func(model,style, content,style)
        fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 3 * num_samples))
        [ax.axis("off") for ax in np.ravel(axes)]

        for axis, style_image, content_image, nst_image in zip(
            axes, style[:num_samples], content[:num_samples], reconstructed_image[:num_samples]
        ):
            plot_images(axis, style_image, content_image, nst_image)
        sample_index += 1
        name = f"sample_img_{sample_index}"
        save_path = os.path.join(output_dir, f"{name}.png")
        metrics["name"] = name
        plt.savefig(save_path)
        plt.close(fig)
        plt.show()
        metrics_list.append(metrics)
