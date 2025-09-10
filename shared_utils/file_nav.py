import os

def get_base_name(file_path : str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]

def get_image_files(folder_path : str,image_file_types : tuple[str ]=('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')) -> list[str]:
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_file_types)]