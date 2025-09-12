import keras
from shared_utils.gatys_network import get_style_layer_names, get_content_layer_names
from shared_utils.network import get_model_for_loss_net
def get_model(model_name : str = "vgg19",img_width : int = 224,img_height : int = 224,use_model_layers = True,config_layers = {}):
  """ Creates our model with access to intermediate layers. 
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
  Returns:
    returns a keras model that takes image inputs and outputs the style and 
      content intermediate layers. 
  """
  # Load our model. We load pretrained VGG, trained on imagenet data (weights=’imagenet’)
  vgg = get_model_for_loss_net(model_name,image_size=(img_height,img_width))
  vgg.trainable = False
  
  # Get output layers corresponding to style and content layers 
  if use_model_layers:
     model_outputs = dict([(layer.name, layer.output) for layer in vgg.layers])
  else:
    style_layer_names = config_layers.get("style", get_style_layer_names(model_name))
    content_layer_names = config_layers.get("Content", get_content_layer_names(model_name)) 
    style_outputs = {name: vgg.get_layer(name).output for name in style_layer_names}
    content_outputs = {name: vgg.get_layer(name).output for name in content_layer_names}
    model_outputs = {**style_outputs, **content_outputs}


  # Build model 

  return keras.Model(vgg.input, model_outputs)