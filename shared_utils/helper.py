import os
def create_dir(folder_path: str):
     if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_save_path(top_folder_name = "scream", model_save_path = "../content/drive/MyDrive/Models", enable_versions = False,resume = 1):
    os.makedirs(model_save_path, exist_ok=True)
    save_path = os.path.join(model_save_path, top_folder_name)
    version = 1
    final_top_folder_name = top_folder_name
    if enable_versions:
        while os.path.exists(save_path):
            final_top_folder_name = f"{top_folder_name}_v{version}"
            save_path = os.path.join(model_save_path, f"{final_top_folder_name}")
            version += 1
    print("Save Path: ", save_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path, final_top_folder_name