def convert_to_numpy(d):
    metrics = ["content_loss", "style_loss", "total_variation_loss"]
    for x in metrics:
        d[x] = [y.numpy() for y in d[x] if hasattr(y, 'numpy')]

    return d