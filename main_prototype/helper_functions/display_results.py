import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from requests import get
import matplotlib.backends.backend_cairo
def open_file(style_path):
   
    img = Image.open(style_path)
    img = np.array(img)
    return img

def display_image(img):
    plt.imshow(img)
    plt.axis("off")

def display_use_NST_img(get_image_paths,config):   
    content_path, style_path = get_image_paths

    content_img = open_file(content_path)
    style_img = open_file(style_path)

    plt.figure(figsize=(6, 6))
    # content 
    plt.subplot(4, 3, 1)
    display_image(content_img)
    plt.title("Content image")

    # style
    plt.subplot(4, 3, 2)
    display_image(style_img)
    plt.title("Style image")
    plt.tight_layout()
    plt.show()
    optimizer = config.get('optimizer', '')
    loss_network = config.get('ln', '')
    lr = config.get('lr', '')
    plt.text(0.5, -0.1, f"Optimizer: {optimizer}, Loss Network: {loss_network}, Learning Rate: {lr}", ha='center', fontsize=12)


def display_NST_results(generated_images, best_image, iterations, losses, get_image_paths,step_range = range(0,10),config = {}):
    display_use_NST_img(get_image_paths,config)
    plt.figure(figsize=(12, 12))
    start_index = 0
    plot_start = 1
    include = config.get('include_pics', True)
    if include:
        for i in step_range:
            plt.subplot(4, 3, i + plot_start)
            display_image(generated_images[i + start_index])
            plt.title(f"Loss: {losses[i + start_index]:.2f}, Iterations: {iterations[i + start_index]}", fontsize=10)
        plt.show()


    plt.figure(figsize=(8, 8))
    display_image(best_image.get_image())
    plt.title(f"Best Image\nloss: {best_image.get_cost():.2f}, Iterations: {best_image.get_iterations()}", fontsize=10)
    
    plt.savefig("best_image.png")
    plt.show()
    

def save_cap(cap,get_image_paths,prefix=""):
    if get_image_paths is None:
        return
    if get_image_paths is None or len(get_image_paths) < 2:
        return
    save_dir = "evaluation_images"
    os.makedirs(save_dir, exist_ok=True)

    content_path, style_path = get_image_paths
    content_name : str  = os.path.splitext(os.path.basename(content_path))[0]
    style_name : str  = os.path.splitext(os.path.basename(style_path))[0]
    file_name : str = f"{prefix}_{content_name}_{style_name}"
    save_plt_path = os.path.join(save_dir, f"{file_name}.png")
    with open(save_plt_path, 'w') as file:
        file.write(cap.stdout)