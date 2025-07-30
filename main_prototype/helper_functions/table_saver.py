import matplotlib.pyplot as plt
import pandas as pd
def save_data_logs(image_data_logs, image_paths):
    for i,x in enumerate(image_data_logs):
        df = pd.DataFrame(data=x)
        content_name = image_paths[i][0]
        style_name = image_paths[i][1]
        df.to_csv(f"{content_name}_{style_name}_image_data_logs.csv", index=False)
    
def plot_losses(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['iterations'], df['loss'], label='Loss')
    plt.title(f"Loss over iterations")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_NST_losses(df, filter = []):
    plt.figure(figsize=(10, 5))
    for key,value in df["metrics"].items():
        plt.plot(df['iterations'], value, label=f'{key} loss')
    plt.title(f"NST Losses over iterations")
    plt.xlabel('Iterations')
    plt.ylabel('NST Losses')
    plt.legend()
    plt.show()