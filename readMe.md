# StyleMotion - A Model with a Interactive User Interface for Filmmakers

## Overview

StyleMotion is a real-time neural style transfer model that supports real-time, multi-neural style transfer, image-based and video-based style transfer. It can process videos up tp 900\*900 resolution.

Neural style transfer is a process where two images: content and style image are supplied
and the style of the second image is applied the the content image, generating a new image.

## Models

### AdaIN Model (StyleMotion)

The StyleMotion model is based on the [ReCoNet paper](https://arxiv.org/abs/1807.01197) and the [AdaIN paper](https://arxiv.org/abs/1703.06868).

### Gatys et al. Model

The Gatys model is based on the [Gatys et al. paper](https://arxiv.org/abs/1508.06576). This uses a different optimizer to original project (using ADAM rather than SGD).
This model also supports a simple version of multi-neural style transfer.

### Johnson et al. Model

The Johnson model is based on the [Johnson et al. paper](https://arxiv.org/abs/1603.08155).

## Huang et al. Model

The Huang model is based on the [Huang et al. paper](https://arxiv.org/abs/1703.06868).

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

The Huang model is stored in the 'Huang/' directory and its training is handle in the 'training.py' file.

The Johnson al et model is stored in the 'forward_feed/' directory and its training is handle in the 'training.py' file.

## Code

### shared utils

The 'shared_utils' directory is used to hold code that each of the models will used such as loss functions.

#### files

'device.py' holds code for getting hardware metrics such as 'RAM', 'CPU' and 'GPU' from the model.

## Steps for running the code

1. Go into the respective directory for each model.
2. Create a virtual environment in.
3. Use pip install -r requirements.txt to download the dependencies required.
4. Go to the notebook.

## Model Results

### AdaIN Model (StyleMotion)

### Gatys et al. Model

### Johnson et al. Model

## Huang et al. Model
