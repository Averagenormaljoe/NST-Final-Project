from calendar import c
from curses import meta
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import pandas as pd
import os

from requests import get

from main_prototype.helper_functions.meta_manager import prepare_metadata, save_metadata
from shared_utils.file_nav import get_base_name 




def save_data_logs(image_data_logs, image_paths,folder = "csv_hardware"):

    os.makedirs(folder, exist_ok=True)
    for i,x in enumerate(image_data_logs):
        df = pd.DataFrame(data=x)
        content_name = get_base_name(image_paths[i][0])
        style_name = get_base_name(image_paths[i][1])
        name = f"{content_name}_{style_name}_image_data_logs"
        output_path = os.path.join(folder, name)
        df.to_csv(f"{output_path}.csv", index=False)
        avg = get_avg_metrics(df)
        config = {
            "content_name": content_name,
            "style_name": style_name
        }
        metadata = prepare_metadata(
            config=config,
            file_paths=image_paths[i],
            extra_dict=avg
        )
        metadata_file_name = f"{output_path}_metadata.json"
        save_metadata(metadata, metadata_file_name)
        
                
def get_avg_metrics(df):
    metrics_to_avg = ['ssim', 'psnr', 'lpips', 'total_variation', 'content', 'style', 'ms_ssim']
    avg = {}
    for metric in metrics_to_avg:
        if metric in df.columns:
            avg_value = df[metric].mean()
            avg[f"avg_{metric}"] = avg_value
    return avg
        
    
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