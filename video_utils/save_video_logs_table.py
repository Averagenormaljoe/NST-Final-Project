import os
import pandas as pd
from video_utils.convert_to_numpy import convert_to_numpy
def save_video_logs_table(logs : list, save_path : str, prefix : str = "logs_video") -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    enumerate_logs = enumerate(logs)
    for i, x in enumerate_logs:
        df = pd.DataFrame(x)
        df['temporal_loss'] = df['temporal_loss'].apply(lambda x: x.numpy() if hasattr(x, 'numpy') else x)
        filename = f"{prefix}_{i}.csv"
        csv_path = os.path.join(save_path, filename)
        df.to_csv(csv_path, index=False)