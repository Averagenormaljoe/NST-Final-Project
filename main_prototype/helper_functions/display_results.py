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
    return ax

def display_meta_plt(fig : Figure,config) -> Figure:
    optimizer = config.get('optimizer', '')
    loss_network = config.get('ln', '')
    lr = config.get('lr', '')
   
    ax_meta = fig.add_subplot(5, 3, 3)
    ax_meta.axis('off')
    ax_meta.text(0.5, -0.1, f"Optimizer: {optimizer}, Loss Network: {loss_network}, Learning Rate: {lr}", ha='center', fontsize=12)
    return fig

def display_NST_img(fig : Figure, img, name = "img",index = 1) -> Figure:
    ax = fig.add_subplot(5, 3, index)
    ax = display_image(ax,img)
    ax.set_title(name)
    ax.axis('off')
    return fig

def display_use_NST_img(fig : Figure,image_file_paths,config) -> Figure:   
    content_path, style_path = image_file_paths
    content_img = open_file(content_path)
    style_img = open_file(style_path)
    # content 
    fig = display_NST_img(fig, content_img, name="Content Image",index=1)
    # style
    fig = display_NST_img(fig, style_img, name="Style Image",index=2)
    # meta
    fig = display_meta_plt(fig,config)
    

    return fig
def display_loss_img(fig : Figure, generated_images, losses : list[float],i : int, iterations : list[int],plot_start : int, start_index : int = 0):
    ax = fig.add_subplot(5, 3, i + plot_start)
    display_image(ax,generated_images[i + start_index])
    ax.set_title(f"Loss: {losses[i + start_index]:.2f}, Iterations: {iterations[i + start_index]}", fontsize=10)
    ax.axis('off')

def display_NST_results(generated_images, best_image, iterations, losses, image_file_paths,start_index = 0,config = {}):

    save_dir = "evaluation_images"
    os.makedirs(save_dir, exist_ok=True)
    
    content_path, style_path = image_file_paths
    content_name : str  = os.path.splitext(os.path.basename(content_path))[0]
    style_name : str  = os.path.splitext(os.path.basename(style_path))[0]
    fig = plt.figure(figsize=(16, 18))
    fig = display_use_NST_img(fig,image_file_paths,config)
    plot_start = 4
    include = config.get('include_pics', True)
    img_range = range(start_index, start_index + 10)
    if include:
        for i in img_range:
            display_loss_img(fig, generated_images, losses, i, iterations, plot_start, start_index)
    show_best = config.get('show_best', True)
    if show_best:
        ax_best = fig.add_subplot(5, 3, 3)
        ax_best = display_image(ax_best,best_image.get_image())
        ax_best.set_title(f"Best Image\nLoss: {best_image.get_cost():.2f}, Iter: {best_image.get_iterations()}")
        ax_best.axis('off')
        show_name = config.get('show_name', False)
        if show_name:
            ax_best.text(0.5, -0.1, f"Content: {content_name}, Style: {style_name}", ha='center', fontsize=12)
    show_plt(fig)
    should_save = config.get('should_save', True)
    if should_save:
        save_fig(fig, image_file_paths, prefix=f"{start_index}")



def show_plt(fig : Figure):
    plt.tight_layout()
    if fig is None:
        return
    plt.show(fig)
    plt.close(fig)

def save_fig(fig : Figure,image_file_paths,prefix=""):
    if image_file_paths is None:
        return
    if image_file_paths is None or len(image_file_paths) < 2:
        return
    save_dir = "evaluation_images"
    os.makedirs(save_dir, exist_ok=True)

    content_path, style_path = image_file_paths
    content_name : str  = os.path.splitext(os.path.basename(content_path))[0]
    style_name : str  = os.path.splitext(os.path.basename(style_path))[0]
    file_name : str = f"{prefix}_{content_name}_{style_name}"
    save_plt_path = os.path.join(save_dir, f"{file_name}.png")
    fig.savefig(save_plt_path, bbox_inches='tight', pad_inches=0.1)

