import os
import pandas as pd
from forward_feed.model_functions.convert_to_numpy import convert_to_numpy
def save_logs_table(logs, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    enumerate_logs = enumerate(logs)

    for i, x in enumerate_logs:
        x = convert_to_numpy(x)
        df = pd.DataFrame(x)
        filename = f"logs_epoch_{i}.csv"
        csv_path = os.path.join(save_path, filename)
        df.to_csv(csv_path, index=False)