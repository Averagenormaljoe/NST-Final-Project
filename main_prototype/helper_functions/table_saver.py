import matplotlib.pyplot as plt
import pandas as pd
import os

from helper_functions.meta_manager import prepare_metadata, save_metadata
from shared_utils.file_nav import get_base_name 




def save_data_logs(image_data_logs, image_paths,folder = "csv_hardware"):

    os.makedirs(folder, exist_ok=True)
    for i,x in enumerate(image_data_logs):
        df = pd.DataFrame(data=x)
        content_path, style_path = image_paths[i]
        content_name = get_base_name(content_path)
        style_name = get_base_name(style_path)
        name = f"{content_name}_{style_name}_image_data_logs"
        output_path = os.path.join(folder, name)
        df.to_csv(f"{output_path}.csv", index=False)
        avg = get_avg_metrics(df)
        total_time_dict = get_total_time(df)
        avg.update(total_time_dict)
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
        plot_losses(df, image_paths[i])
        plot_NST_losses(df, image_paths[i])
        
                
def get_avg_metrics(df):
    metrics_to_avg = ['ssim',  'lpips', 'total_variation', 'content', 'style', 'ms_ssim']
    avg = {}
    for metric in metrics_to_avg:
        if metric in df.columns:
            avg_value = df[metric].mean()
            avg[f"avg_{metric}"] = avg_value
    return avg
def get_total_time(df):
    times = ["cpu time", "gpu time"]
    time_dict = {}
    for time in times:
        if time in df.columns:
            total_time = df[time].sum()
            time_dict[f"total_{time}"] = total_time
    return time_dict
     
     
def show_or_close_plot(show_plot):
    if show_plot:
        plt.show()
    else:
        plt.close()   
    
def plot_losses(df,image_paths,show_plot=True):
    plt.figure(figsize=(10, 5))
    plt.plot(df['iterations'], df['loss'], label='Loss')
    plt.title(f"Loss vs iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plot_name = "default"
    save_plot(image_paths, plot_name)
    show_or_close_plot(show_plot)
    
    

def save_plot(image_paths,name, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    content_path, style_path = image_paths
    content_name = get_base_name(content_path)
    style_name = get_base_name(style_path )
    save_name = f"{content_name}_{style_name}_{name}_loss_plot.png"
    save_path = os.path.join(folder, save_name)
    plt.savefig(save_path)

def plot_NST_losses(df,image_paths, show_plot=True,verbose=True):
    metric_keys = ['ssim', 'lpips', 'ms_ssim']
    plt.figure(figsize=(10, 5))
    not_found_keys = [k for k in metric_keys if k not in df.columns]
    if not_found_keys and verbose:
        print(f"Error: keys ({not_found_keys}) are missing from the dataframe")
        return
    for key in metric_keys:
        plt.plot(df['iterations'], df[key], label=f'{key} loss')
    plt.title(f"NST Losses vs iterations")
    plt.xlabel('Iterations')
    plt.ylabel('NST Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_name = "NST"
    save_plot(image_paths,plot_name)
    show_or_close_plot(show_plot)
