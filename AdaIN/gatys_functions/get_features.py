def get_features(features, layer_name,base_index, combination_index):
      layer_features = features[layer_name]
      base_features = layer_features[base_index, :, :, :]
      combination_features = layer_features[combination_index, :, :, :]
      return base_features, combination_features