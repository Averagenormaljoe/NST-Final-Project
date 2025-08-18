import os
import subprocess   

def accept_url(url):
    subprocess.run(url.split(), check=True)

def download_monkaa_dataset():
    if os.path.exists("monkaa.zip"):
        print("Monkaa dataset already downloaded.")
        return
    url = "wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar"
    accept_url(url)
    
def flying_things_3d():
    if os.path.exists("FlyingThings3D.zip"):
        print("Flying Things 3D dataset already downloaded.")
        return
    url = "wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar.torrent"
    accept_url(url)

def download_driving_dataset():
    if os.path.exists("driving_dataset.zip"):
        print("Driving dataset already downloaded.")
        return
    url = "wget https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Driving/raw_data/driving__frames_cleanpass.tar.torrent"
    accept_url(url)

