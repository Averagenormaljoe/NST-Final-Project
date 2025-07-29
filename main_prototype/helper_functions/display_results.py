import os
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
def open_file(style_path):
   
    img = Image.open(style_path)
    img = np.array(img)
    return img

def display_image(ax : Axes,img):
    ax.imshow(img)
    ax.axis('off')

def display_use_NST_img(fig : Figure,image_paths,config):   
    content_path, style_path = image_paths

    content_img = open_file(content_path)
    style_img = open_file(style_path)
    # content 
    ax1 = fig.add_subplot(4, 3, 1)
    display_image(ax1,content_img)
    ax1.set_title("Content image")
    # style
    ax2 = fig.add_subplot(4, 3, 2) 
    display_image(ax2,style_img)
    ax2.set_title("Style image")
    
    optimizer = config.get('optimizer', '')
    loss_network = config.get('ln', '')
    lr = config.get('lr', '')
    ax_meta = fig.add_subplot(4, 3, 3)
    ax_meta.axis('off')
    ax_meta.text(0.5, -0.1, f"Optimizer: {optimizer}, Loss Network: {loss_network}, Learning Rate: {lr}", ha='center', fontsize=12)


def display_NST_results(generated_images, best_image, iterations, losses, image_paths,config = {}, img_range = range(0, 10)):

    save_dir = "evaluation_images"
    os.makedirs(save_dir, exist_ok=True)
    
    content_path, style_path = image_paths
    content_name : str  = os.path.splitext(os.path.basename(content_path))[0]
    style_name : str  = os.path.splitext(os.path.basename(style_path))[0]
    file_name : str = f"{content_name}_{style_name}"
    fig = plt.figure(figsize=(16, 18))
    display_use_NST_img(fig,image_paths,config)
    
    start_index = 0
    plot_start = 1
    include = config.get('include_pics', True)
    if include:
        for i in img_range:
            ax = fig.add_subplot(4, 3, i + plot_start)
            display_image(ax,generated_images[i + start_index])
            ax.set_title(f"Loss: {losses[i + start_index]:.2f}, Iterations: {iterations[i + start_index]}", fontsize=10)
            ax.axis('off')

    ax_best = fig.add_subplot(5, 3, 14)
    display_image(ax_best,best_image.get_image())
    ax_best.set_title(f"Best Image\nLoss: {best_image.get_cost():.2f}, Iter: {best_image.get_iterations()}")
    ax_best.axis('off')
    plt.tight_layout()
    save_plt_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(save_plt_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.show()