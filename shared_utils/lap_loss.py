import cv2
import numpy as np
import tensorflow as tf
def lap_loss(base_img, stylized_img):

    base_lap_tf = apply_lap_process(base_img)
    stylized_lap_tf = apply_lap_process(stylized_img)


    loss_fn = tf.keras.losses.MeanSquaredError()
    return loss_fn(base_lap_tf, stylized_lap_tf)

def apply_lap_process(img):
    
    
    numpy_img = np.asarray(img, dtype=np.float64)
    base_lap = cv2.Laplacian(numpy_img, ddepth=cv2.CV_64F, ksize=3)

    base_lap_tf = tf.convert_to_tensor(base_lap, dtype=tf.float32)

    return base_lap_tf 