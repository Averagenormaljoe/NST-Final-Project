import matplotlib.pyplot as plt
import pandas as pd
import os

from shared_utils.file_nav import get_base_name 

import json


def save_data_logs(image_data_logs, image_paths,folder = "csv_hardware"):
    
    os.makedirs(folder, exist_ok=True)
    for i,x in enumerate(image_data_logs):
        df = pd.DataFrame(data=x)
        content_name = get_base_name(image_paths[i][0])
        style_name = get_base_name(image_paths[i][1])
        name = f"{content_name}_{style_name}_image_data_logs"
        output_path = os.path.join(folder, name)
        df.to_csv(f"{output_path}.csv", index=False)
        
    
def plot_losses(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['iterations'], df['loss'], label='Loss')
    plt.title(f"Loss over iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_NST_losses(df, metrics_filter = []):
    metric_keys = ['content', 'style', 'ssim', 'psnr', 'total_variation']
    if metrics_filter:
        metric_keys = [k for k in metric_keys if k in metrics_filter]
    plt.figure(figsize=(10, 5))
    missing_keys = [k for k in metric_keys if k not in df.columns]
    if missing_keys:
        print(f"Error: keys ({missing_keys}) are missing from the dataframe")
        return
    for key in metric_keys:
        plt.plot(df['iterations'], df[key], label=f'{key} loss')
    plt.title(f"NST Losses over iterations")
    plt.xlabel('Iterations')
    plt.ylabel('NST Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()