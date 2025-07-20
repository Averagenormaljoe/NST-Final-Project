from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def open_file(style_path):
   
    img = Image.open(style_path)
    img = np.array(img)
    return img

def display_image(img):
    plt.imshow(img)
    plt.axis("off")

def display_use_NST_img(image_paths,config):   
    content_path, style_path = image_paths

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
    plt.text(0.5, -0.1, f"Optimizer: {optimizer}, Loss Network: {loss_network}", ha='center', fontsize=12)


def display_NST_results(generated_images, best_image, iterations, losses, image_path,config = {}):
    display_use_NST_img(image_path,config)
    plt.figure(figsize=(12, 12))
    start_index = 0
    num = len(generated_images)
    plot_start = 1
    for i in range(num):
        plt.subplot(4, 3, i + plot_start)
        display_image(generated_images[i + start_index])
        plt.title(f"Loss: {losses[i + start_index]:.2f}, Iterations: {iterations[i + start_index]}", fontsize=10)
    plt.show()


    plt.figure(figsize=(8, 8))
    display_image(best_image.get_image())
    plt.title(f"Best Image\nloss: {best_image.get_cost():.2f}, Iterations: {best_image.get_iterations()}", fontsize=10)
    
    plt.show()