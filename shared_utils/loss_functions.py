import tensorflow as tf
def content_loss(base_img, combination_img):
 assert base_img.shape == combination_img.shape
 return tf.reduce_sum(tf.square(combination_img - base_img))


def mean_style_loss(style_img, combination_img):
    axes = [1, 2]
    style_mean = tf.reduce_mean(style_img, axis=axes)
    combination_mean = tf.reduce_mean(combination_img, axis=axes)
    return tf.reduce_mean(tf.square(style_mean - combination_mean))

def gram_matrix(x):
 x = tf.transpose(x, (2, 0, 1))
 features = tf.reshape(x, (tf.shape(x)[0], -1))
 gram = tf.matmul(features, tf.transpose(features))
 return gram

def equal_blends(gram_matrices):
    return sum(gram_matrices) / len(gram_matrices)

def style_loss(style_img, combination_img,img_width : int,img_height: int):
 assert style_img.shape == combination_img.shape
 S = gram_matrix(style_img)
 C = gram_matrix(combination_img)
 channels = 3
 size = img_height * img_width
 diff_squared = tf.square(S - C)
 sum_diff = tf.reduce_sum(diff_squared)
 sum_norm = sum_diff / (4.0 * (channels ** 2) * (size ** 2))
 return sum_norm 


def extract_image_info(x,img_width,img_height):
    a = tf.square(
    x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1, :]
    )
    b = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :]
    )
    return a,b

def high_pass_x_y(image, size = None):
    if size:
        x,y = extract_image_info(image,size[0],size[1])
    else :
        x = image[:, :, 1:, :] - image[:, :, :-1, :]
        y = image[:, 1:, :, :] - image[:, :-1, :, :]         
    return x, y

def total_variation_loss(x,tv_type="gatys" ,size = None,):
    a,b = high_pass_x_y(x, size)
    if tv_type == "gatys":
        return total_variation_loss_gatys(a,b)
    elif tv_type == "l1":
        return total_variation_loss_l1(a,b)
    elif tv_type == "l2":
        return total_variation_loss_l2(a,b)
    
def total_variation_loss_gatys(a,b):
    return tf.reduce_sum(tf.pow(a + b, 1.25))
def total_variation_loss_l1(a,b):
        return tf.reduce_sum(tf.abs(a) + tf.abs(b))

def total_variation_loss_l2(a,b):
    return tf.reduce_sum(tf.square(a) + tf.square(b))


    
