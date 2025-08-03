import os
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE
from AdaIN_functions.helper import decode_and_resize,extract_image_from_voc
import tensorflow_datasets as tfds
# Defining the global variables.
BATCH_SIZE = 64



def load_train_dataset(dataset_use: str,train_style, batch_size : int,num_parallel_calls : int):
    train_style_ds = (
    tf.data.Dataset.from_tensor_slices(train_style)
    .map(decode_and_resize, num_parallel_calls=num_parallel_calls)
    .repeat()
    ) 
    
    train_content_ds = tfds.load(dataset_use, split='train').map(extract_image_from_voc).repeat()
    return train_style_ds, train_content_ds


def save_train_dataset(dataset_path,dataset_use: str, train_style, should_save: bool = True):
    if os.path.exists(dataset_path):
        train_style_ds = tf.data.Dataset.load(f"{dataset_path}/train_style_ds")
        train_content_ds = tf.data.Dataset.load(f"{dataset_path}/train_content_ds")
    else:
        train_style_ds,train_content_ds = load_train_dataset(dataset_use, train_style,BATCH_SIZE,AUTOTUNE)
        if should_save:
            train_style_ds.save(f"{dataset_path}/train_style_ds")
            train_content_ds.save(f"{dataset_path}/train_content_ds")
    return train_style_ds, train_content_ds


def load_val_test_datasets(dataset_use: str, val_style, test_style, batch_size: int,num_parallel_calls : int):
    val_style_ds = (
        tf.data.Dataset.from_tensor_slices(val_style)
        .map(decode_and_resize, num_parallel_calls=num_parallel_calls)
        .repeat()
    )
    val_content_ds = (
        tfds.load(dataset_use, split="validation").map(extract_image_from_voc).repeat()
    )

    test_style_ds = (
        tf.data.Dataset.from_tensor_slices(test_style)
        .map(decode_and_resize, num_parallel_calls=num_parallel_calls)
        .repeat()
    )
    test_content_ds = (
        tfds.load(dataset_use, split="test")
        .map(extract_image_from_voc, num_parallel_calls=num_parallel_calls)
        .repeat()
    )

    val_ds = (
        tf.data.Dataset.zip((val_style_ds, val_content_ds))
        .shuffle(batch_size * 2)
        .batch(batch_size)
        .prefetch(num_parallel_calls)
    )

    test_ds = (
        tf.data.Dataset.zip((test_style_ds, test_content_ds))
        .shuffle(batch_size * 2)
        .batch(batch_size)
        .prefetch(num_parallel_calls)
    )
    

 


    return val_ds, test_ds

def get_train_ds(train_style_ds, train_content_ds, batch_size,autotune):
    # Zipping the style and content datasets.
    train_ds = (
        tf.data.Dataset.zip((train_style_ds, train_content_ds))
        .shuffle(batch_size * 2)
        .batch(batch_size)
        .prefetch(autotune)
    )
    return train_ds