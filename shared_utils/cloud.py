import os
import sys


def colab_paths():
    sys.path.append('/content/drive/MyDrive')
    nb_path = '/content/mnt/MyDrive/Library'
    os.makedirs(nb_path, exist_ok=True)
    sys.path.insert(0,nb_path)
    
def kaggle_paths(kaggle_script_paths,kaggle_dir_path):
    sys.path.insert(1, kaggle_dir_path)
    for x in  kaggle_script_paths:
      if x not in sys.path:
        sys.path.insert(1, x)
        
