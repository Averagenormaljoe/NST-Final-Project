import os
import pandas as pd
from video_utils.convert_to_numpy import convert_to_numpy
def save_video_logs_table(logs, save_path, prefix = "logs_video"):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    enumerate_logs = enumerate(logs)
    for i, x in enumerate_logs:
        x = convert_to_numpy(x,["temporal_loss"])
        df = pd.DataFrame(x)
        filename = f"{prefix}_{i}.csv"
        csv_path = os.path.join(save_path, filename)
        df.to_csv(csv_path, index=False)