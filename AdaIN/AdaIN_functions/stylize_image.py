from AdaIN_functions.ada_in import ada_in
def stylize_image(model, content_image, style_image):
    """Stylizes the content image using the style image."""
    content_encoded = model.encoder(content_image)
    style_encoded = model.encoder(style_image)
    t = ada_in(style=style_encoded, content=content_encoded)
    reconstructed_image = model.decoder(t)
    return reconstructed_image