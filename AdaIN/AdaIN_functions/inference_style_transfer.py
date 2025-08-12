import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
def inference_style_transfer(model, dataset,stylize_func, num_samples=10,output_dir="stylized_images",batch = 1):
    os.makedirs(output_dir, exist_ok=True)
    sample_index = 0
    sample_dataset = dataset if batch == 0 else dataset.take(batch)
    for style, content in sample_dataset:
        reconstructed_image = stylize_func(model,style, content)
        fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 3 * num_samples))
        [ax.axis("off") for ax in np.ravel(axes)]

        for axis, style_image, content_image, nst_image in zip(
            axes, style[:num_samples], content[:num_samples], reconstructed_image[:num_samples]
        ):
            ax_style, ax_content, ax_reconstructed = axis
            ax_style.imshow(style_image)
            ax_style.set_title("Style Image")
            ax_content.imshow(content_image)
            ax_content.set_title("Content Image")
            ax_reconstructed.imshow(nst_image)
            ax_reconstructed.set_title("NST Image")
        sample_index += 1
        save_path = os.path.join(output_dir, f"stylized_image_sample_{sample_index}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        plt.show()
        
    command = f"!zip -r /{output_dir}.zip /{output_dir}"
    split_command = command.split()
    subprocess.run(split_command, shell=False, check=True)