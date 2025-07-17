# StyleMotion - A Model with a Interactive User Interface for Filmmakers

## Overview

StyleMotion is a real-time neural style transfer model that supports real-time, multi-neural style transfer, image-based and video-based style transfer. It can process videos up tp 900\*900 resolution.
This model is complemented with three other models for comparison purposes (Gatys, Johnson and Huang).

Neural style transfer is a process where two images: content and style image are supplied
and the style of the second image is applied the the content image, generating a new image.

## Models

### AdaIN Model (StyleMotion)

The StyleMotion model is based on the [ReCoNet paper](https://arxiv.org/abs/1807.01197) and the [AdaIN paper](https://arxiv.org/abs/1703.06868). This supports video and multi-neural style transfer.

### Gatys et al. Model

The Gatys model is based on the [Gatys et al. paper](https://arxiv.org/abs/1508.06576). This uses a different optimizer to original project (using ADAM rather than SGD).
This model also supports a simple version of multi-neural style transfer.

### Johnson et al. Model

The Johnson model is based on the [Johnson et al. paper](https://arxiv.org/abs/1603.08155).

## Huang et al. Model

The Huang model is based on the [Huang et al. paper](https://arxiv.org/abs/1703.06868). This model also adapt the luminance constraint from the ReCoNet paper

## Streamlit Website

This uses the models created in the project and supplied them to an user interface for users to interact with. This is setup using the Streamlit library Currently, the code for this is stored on
another repository called: [https://stylemotion-app.streamlit.app/]("https://github.com/Averagenormaljoe/NST-Streamlit-website").

You can visit the website, which is running from this link:

[https://stylemotion-app.streamlit.app/](https://stylemotion-app.streamlit.app/).

Note if the project is asleep due to inactivity, press the 'Yes, get this app back up!' button to
restart it.

## Model locations

AdaIN is stored in the 'AdaIN/' directory and its training is handle in the 'stylemotion_adain.ipynb'.

The gatys model is stored in the 'main_protoype/' directory and its training is handle in the 'prototype.ipynb' file

The Huang model is stored in the 'Huang/' directory and its training is handle in the 'training.py' file. This also store methods such as the luminance constraint proposed by the ReCoNet paper.

The Johnson al et model is stored in the 'forward_feed/' directory and its training is handle in the 'training.py' file.

## Code

## General code

'setup.py' exists for compatibility purposes for the pyproject.toml file in each directory of the project.

### StyleMotion (AdaIN) - "AdaIN" directory

The main StyleMotion prototype. Code is adapted from tensorflow website [2].

### Ruber - "Ruber" directory

'main.ipynb' file is the main entry point notebook for the folder.

### main_prototype (Gatys) - "main_prototype" directory

'prototype.ipynb' file is the main entry point notebook for the folder. Code is adapted from Chollet F book [1].

### forward_feed (Johnson) - "forward_feed" directory

'main.ipynb' file is the main entry point notebook for the folder.

### shared utils

The 'shared_utils' directory is used to hold code that each of the models will used such as loss functions.

#### files

The 'device.py' file holds code for getting hardware metrics such as 'RAM', 'CPU' and 'GPU' from the model.

The 'gatys_network.py' file holds code to return the network used by the Gatys model.

## Steps for running the code

### Prerequisites

Note this application was primarily tested in Windows, meaning that MacOS and Linux operations may work differently.

1. Python 3.11 Download and install from python.org. This can be verify with 'python --version'. [Python Official Website](https://www.python.org/)
2. pip (Python package manager), which comes with Python 3.11. This can be verify with 'pip --version'.
3. run 'pip install virtualenv' in a terminal to install the 'virtualenv' library to able to create a virtual environment.
4. Running this in an IDE (VScode or PyCharm) or JupyterLabs is the recommended option for running these notebooks.

### Instructions

1. Go into the respective directory for each model. Use the 'cd [directory_name]' command for to navigate to the directory  
   in the terminal.
2. Create a virtual environment in by running 'python -m venv venv' or 'virtualenv venv' in the terminal.
3. activate the environment with:

'venv\Scripts\activate' (Windows) or 'source venv\Scripts\activate' (MacOS and Linux).

'source venv/Scripts/activate' (Win&Bash).

4. Use pip install -r requirements.txt to download the dependencies required.
5. Go to the notebook. For main_prototype, it is the 'prototype.ipynb' file.
6. Select the 'run all' command at the top of the notebook
7. To deactivate the environment run:

'deactivate'

## Model Results

### AdaIN Model (StyleMotion)

#### Portfolio

#### Table

### Gatys et al. Model

#### Portfolio

#### Table

### Johnson et al. Model

#### Portfolio

#### Table

### Huang et al. Model

#### Portfolio

#### Table

## To-Do

Adding Ruber and Johnson model to the project.

## References

1. F. Chollet. 2018. Deep Learning with Python. Manning Publications.
2. A. R. Gosthipaty and R. Raha. 2021. Neural style transfer with AdaIN. Keras Documentation. Retrieved June 16, 2025 from https://keras.io/examples/generative/adain
