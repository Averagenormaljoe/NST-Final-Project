#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 

# # Setup
# 
# We begin with importing the necessary packages. We also set the
# seed for reproducibility. The global variables are hyperparameters
# which we can change as we like.

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import os
import sys
on_kaggle : bool = True if any("kaggle" in path for path in sys.path) else False
# Defining the global variables.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
# Training for single epoch for time constraint.
# Please use atleast 30 epochs to see good results.
EPOCHS = 1
AUTOTUNE = tf.data.AUTOTUNE


# In[ ]:


from google.colab import files
def upload_files(on_kaggle : bool = True):
    if not on_kaggle and not os.path.exists("kaggle.json"):
        files.upload()
upload_files(on_kaggle)


# In[ ]:


if not on_kaggle:
    get_ipython().system('mkdir ~/.kaggle')
    get_ipython().system('cp kaggle.json ~/.kaggle/')
    get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
    get_ipython().system('kaggle datasets download ikarus777/best-artworks-of-all-time')
    get_ipython().system('unzip -qq best-artworks-of-all-time.zip')
    get_ipython().system('rm -rf images')
    get_ipython().system('mv resized artwork')
    get_ipython().system('rm best-artworks-of-all-time.zip artists.csv')


# In[5]:


def decode_and_resize(image_path):
    """Decodes and resizes an image from the image file path.

    Args:
        image_path: The image file path.

    Returns:
        A resized image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


def extract_image_from_voc(element):
    """Extracts image from the PascalVOC dataset.

    Args:
        element: A dictionary of data.

    Returns:
        A resized image.
    """
    image = element["image"]
    image = tf.image.convert_image_dtype(image, dtype="float32")
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


# In[ ]:


def retrieve_style_image(images_path: str):
    style_images = os.listdir(images_path)
    style_images = [os.path.join(images_path, path) for path in style_images]
    return style_images

# Get the image file paths for the style images.
def get_style_images(on_kaggle : bool = True):
    if on_kaggle:
        style_images = retrieve_style_image("/kaggle/input/best-artworks-of-all-time/resized/resized")
        return style_images
    else:
        style_images = retrieve_style_image("/content/artwork/resized")
        return style_images


# In[ ]:


style_images = get_style_images(on_kaggle)
# split the style images in train, val and test
total_style_images = len(style_images)
train_style = style_images[: int(0.8 * total_style_images)]
val_style = style_images[int(0.8 * total_style_images) : int(0.9 * total_style_images)]
test_style = style_images[int(0.9 * total_style_images) :]


# In[7]:


train_style_ds = (
    tf.data.Dataset.from_tensor_slices(train_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)


# In[ ]:


dataset_options : list[str] = ["clic", "coco/2017"]
chosen_dataset : int = 0
dataset_use : str = dataset_options[chosen_dataset]
train_content_ds = tfds.load(dataset_use, split='train').map(extract_image_from_voc).repeat()


# In[ ]:


val_style_ds = (
    tf.data.Dataset.from_tensor_slices(val_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)
val_content_ds = (
    tfds.load(dataset_use, split="validation").map(extract_image_from_voc).repeat()
)

test_style_ds = (
    tf.data.Dataset.from_tensor_slices(test_style)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .repeat()
)
test_content_ds = (
    tfds.load(dataset_use, split="test")
    .map(extract_image_from_voc, num_parallel_calls=AUTOTUNE)
    .repeat()
)

# Zipping the style and content datasets.
train_ds = (
    tf.data.Dataset.zip((train_style_ds, train_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.zip((val_style_ds, val_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.zip((test_style_ds, test_content_ds))
    .shuffle(BATCH_SIZE * 2)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# ## Visualizing the data
# 
# It is always better to visualize the data before training. To ensure
# the correctness of our preprocessing pipeline, we visualize 10 samples
# from our dataset.

# In[ ]:


def visualize_the_data(train_ds):
    style, content = next(iter(train_ds))
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(5, 30))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (axis, style_image, content_image) in zip(axes, style[0:10], content[0:10]):
        (ax_style, ax_content) = axis
        ax_style.imshow(style_image)
        ax_style.set_title("Style Image")

        ax_content.imshow(content_image)
        ax_content.set_title("Content Image")


# In[ ]:


visualize_the_data(train_ds)


# ### Encoder
# 
# The encoder is a part of the pretrained (pretrained on
# [imagenet](https://www.image-net.org/)) VGG19 model. We slice the
# model from the `block4-conv1` layer. The output layer is as suggested
# by the authors in their paper.

# In[ ]:


def get_encoder():
    vgg19 = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)


    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")


# ### Adaptive Instance Normalization
# 
# The AdaIN layer takes in the features
# of the content and style image. The layer can be defined via the
# following equation:
# 
# ![AdaIn formula](https://i.imgur.com/tWq3VKP.png)
# 
# where `sigma` is the standard deviation and `mu` is the mean for the
# concerned variable. In the above equation the mean and variance of the
# content feature map `fc` is aligned with the mean and variance of the
# style feature maps `fs`.
# 
# It is important to note that the AdaIN layer proposed by the authors
# uses no other parameters apart from mean and variance. The layer also
# does not have any trainable parameters. This is why we use a
# *Python function* instead of using a *Keras layer*. The function takes
# style and content feature maps, computes the mean and standard deviation
# of the images and returns the adaptive instance normalized feature map.

# In[14]:


def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content):
    """Computes the AdaIn feature map.

    Args:
        style: The style feature map.
        content: The content feature map.

    Returns:
        The AdaIN feature map.
    """
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    return t


# ### Decoder
# 
# The authors specify that the decoder network must mirror the encoder
# network.  We have symmetrically inverted the encoder to build our
# decoder. We have used `UpSampling2D` layers to increase the spatial
# resolution of the feature maps.
# 
# Note that the authors warn against using any normalization layer
# in the decoder network, and do indeed go on to show that including
# batch normalization or instance normalization hurts the performance
# of the overall network.
# 
# This is the only portion of the entire architecture that is trainable.

# In[ ]:


from tensorflow_addons.layers import InstanceNormalization



def get_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),
            layers.Conv2D(filters=512, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=128, **config),
            layers.Conv2D(filters=128, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=64, **config),
            layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="tanh",
            ),
        ]
    )
    return decoder


# In[ ]:


def get_loss_net():
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv2", "block5_conv1"]

    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")


# ## Neural Style Transfer
# 
# This is the trainer module. We wrap the encoder and decoder inside
# a `tf.keras.Model` subclass. This allows us to customize what happens
# in the `model.fit()` loop.

# In[ ]:


class NeuralStyleTransfer(tf.keras.Model):
    def __init__(self, encoder, decoder, loss_net, style_weight, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.psnr_tracker = keras.metrics.Mean(name="psnr")
        self.ssim_tracker = keras.metrics.Mean(name="ssim")
        self.grad_norm_tracker = keras.metrics.Mean(name="grad_norm")
        self.tv_loss_tracker = keras.metrics.Mean(name="tv_loss")


    def train_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            # Encode the style and content image.
            style_encoded = self.encoder(style)
            content_encoded = self.encoder(content)

            # Compute the AdaIN target feature maps.
            t = ada_in(style=style_encoded, content=content_encoded)

            # Generate the neural style transferred image.
            reconstructed_image = self.decoder(t)

            # Compute the losses.
            reconstructed_vgg_features = self.loss_net(reconstructed_image)
            style_vgg_features = self.loss_net(style)
            loss_content = self.loss_fn(t, reconstructed_vgg_features[-1])
            for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                    std_inp, std_out
                )
            loss_style = self.style_weight * loss_style
            tv_loss = tf.reduce_mean(tf.image.total_variation(reconstructed_image))
            total_loss = loss_content + loss_style + 1e-6 * tv_loss

        # Compute gradients and optimize the decoder.
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        psnr = tf.image.psnr(content, reconstructed_image, max_val=1.0)
        ssim = tf.image.ssim(content, reconstructed_image, max_val=1.0)
        grad_norm = tf.linalg.global_norm(gradients)


    
        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        
        # new metrics
        self.psnr_tracker.update_state(psnr)
        self.tv_loss_tracker.update_state(tv_loss)
        self.ssim_tracker.update_state(ssim)
        self.grad_norm_tracker.update_state(grad_norm)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "tv_loss": self.tv_loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
            "ssim": self.ssim_tracker.result(),
            "grad_norm": self.grad_norm_tracker.result(),
        }

    def test_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded)

        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(t)

        # Compute the losses.
        recons_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_features[-1])
        for inp, out in zip(style_vgg_features, recons_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style
        tv_loss = tf.reduce_mean(tf.image.total_variation(reconstructed_image))
        total_loss = loss_content + loss_style + 1e-6 * tv_loss
        psnr = tf.image.psnr(content, reconstructed_image, max_val=1.0)
        ssim = tf.image.ssim(content, reconstructed_image, max_val=1.0)

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        # new metrics
        self.tv_loss_tracker.update_state(tv_loss)
        self.psnr_tracker.update_state(psnr)
        self.ssim_tracker.update_state(ssim)
        
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "tv_loss": self.tv_loss_tracker.result(),
            "psnr": self.psnr_tracker.result(),
            "ssim": self.ssim_tracker.result(),
        }
    def call(self, inputs):
        style = inputs[0]
        content = inputs[1]

        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        t = ada_in(style=style_encoded, content=content_encoded)
        reconstructed_image = self.decoder(t)

        return reconstructed_image
    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
            self.tv_loss_tracker,
            self.psnr_tracker,
            self.ssim_tracker,
            self.grad_norm_tracker,
        ]


# ## Train Monitor callback
# 

# In[18]:


test_style, test_content = next(iter(test_ds))


class TrainMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Encode the style and content image.
        test_style_encoded = self.model.encoder(test_style)
        test_content_encoded = self.model.encoder(test_content)

        # Compute the AdaIN features.
        test_t = ada_in(style=test_style_encoded, content=test_content_encoded)
        test_reconstructed_image = self.model.decoder(test_t)

        # Plot the Style, Content and the NST image.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(tf.keras.utils.array_to_img(test_style[0]))
        ax[0].set_title(f"Style: {epoch:03d}")

        ax[1].imshow(tf.keras.utils.array_to_img(test_content[0]))
        ax[1].set_title(f"Content: {epoch:03d}")

        ax[2].imshow(
            tf.keras.utils.array_to_img(test_reconstructed_image[0])
        )
        ax[2].set_title(f"NST: {epoch:03d}")

        plt.show()
        plt.close()


# ## Train the model

# In[19]:


optimizer = keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = keras.losses.MeanSquaredError()

encoder = get_encoder()
loss_net = get_loss_net()
decoder = get_decoder()

model = NeuralStyleTransfer(
    encoder=encoder, decoder=decoder, loss_net=loss_net, style_weight=4.0
)

model.compile(optimizer=optimizer, loss_fn=loss_fn)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=50,
    validation_data=val_ds,
    validation_steps=50,
    callbacks=[TrainMonitor()],
)


# In[ ]:


model.build((None, *IMAGE_SIZE, 3))


# Saving the model. This is to allow the model to used in the Streamlit library.

# In[ ]:


def save_model(model):
    model.encoder.save("encoder")
    model.decoder.save("decoder")
    model.loss_net.save("loss_net")
    model.save("model")


# In[ ]:


save_model(model)
get_ipython().system('zip -r model.zip encoder decoder loss_net model')
get_ipython().system('mv model.zip /content')


# Next, a function is defined to load the model.

# In[ ]:


def load_model():
    """Loads the saved model."""
    encoder = keras.models.load_model("encoder")
    decoder = keras.models.load_model("decoder")
    loss_net = keras.models.load_model("loss_net")
    model = keras.models.load_model("model")
    return encoder, decoder, loss_net, model


# Next, the model will be loaded. 

# In[ ]:


encoder, decoder, loss_net, model = load_model()


# ## Inference
# 
# After we train the model, we now need to run inference with it. We will
# pass arbitrary content and style images from the test dataset and take a look at
# the output images.
# 
# *NOTE*: To try out the model on your own images, you can use this
# [Hugging Face demo](https://huggingface.co/spaces/ariG23498/nst).

# In[ ]:


def inference_style_transfer(model, dataset, num_samples=10):
    for style, content in dataset.take(1):
        style_encoded = model.encoder(style)
        content_encoded = model.encoder(content)
        t = ada_in(style=style_encoded, content=content_encoded)
        reconstructed_image = model.decoder(t)
        fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 3 * num_samples))
        [ax.axis("off") for ax in np.ravel(axes)]

        for axis, style_image, content_image, nst_image in zip(
            axes, style[:num_samples], content[:num_samples], reconstructed_image[:num_samples]
        ):
            ax_style, ax_content, ax_reconstructed = axis
            ax_style.imshow(style_image)
            ax_style.set_title("Style Image")
            ax_content.imshow(content_image)
            ax_content.set_title("Content Image")
            ax_reconstructed.imshow(nst_image)
            ax_reconstructed.set_title("NST Image")
        plt.show()


# In[ ]:


inference_style_transfer(model, test_content, num_samples=10)

