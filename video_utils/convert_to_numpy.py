def convert_to_numpy(d,metrics = ["content_loss", "style_loss", "total_variation_loss"]):
    for x in metrics:
        if x in d:
            d[x] = [y.numpy() for y in d[x] if hasattr(y, 'numpy')]
    d["total_variation_loss"] = sum(d["total_variation_loss"]) if type(d["total_variation_loss"]) == list else d["total_variation_loss"]
    return d