

import os
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE
import tensorflow_datasets as tfds
from fastai.data.external import untar_data
from helper import get_style_images, decode_and_resize,extract_image_from_voc
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

def create_dataset_dir(dataset_dir = 'coco'):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir

def load_train_dataset_coco(batch_size: int, num_parallel_calls: int):
    dataset_dir = create_dataset_dir('coco')
    coco_url = 'http://images.cocodataset.org/zips/train2014.zip'
    if os.path.exists('coco.zip'):
        print("Skipping download as Coco dataset already exists.")
    else:
        untar_data(coco_url, 'coco.zip', dataset_dir)
    train_style_ds = (
        tf.data.Dataset.list_files(os.path.join('coco/train2014', '*.jpg'))
        .map(decode_and_resize, num_parallel_calls=num_parallel_calls)
        .repeat()
    )
    train_content_ds = (
        tf.data.Dataset.list_files(os.path.join('coco/train2014', '*.jpg'))
        .map(decode_and_resize, num_parallel_calls=num_parallel_calls)
        .repeat()
    )
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
    

    # Zipping the style and content datasets.
    train_ds = (
        tf.data.Dataset.zip((train_style_ds, train_content_ds))
        .shuffle(BATCH_SIZE * 2)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


    return val_ds, test_ds, train_ds