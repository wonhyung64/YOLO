#%%
import tensorflow as tf
from typing import Tuple
import tensorflow_datasets as tfds

def prefetch_dataset(datasets, data_num, batch_size):
    train_set, valid_set, test_set = datasets
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int64),
    )

    tf.random.set_seed(42)
    train_set = train_set.map(lambda x: preprocess(x, split="train", img_size=img_size))
    test_set = test_set.map(lambda x: preprocess(x, split="test", img_size=img_size))
    valid_set = valid_set.map(lambda x: preprocess(x, split="validation", img_size=img_size))

    train_set = train_set.shuffle(buffer_size=data_num, seed=42)

    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    test_set = test_set.repeat().padded_batch(
        batch_size=1,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def load_dataset(name, data_dir):
    train1, dataset_info = tfds.load(
        name=name,
        split="train",
        data_dir=data_dir,
        with_info=True
    )
    train2 = tfds.load(
        name=name,
        split="validation[100:]",
        data_dir=data_dir,
    )
    valid_set = tfds.load(
        name=name,
        split="validation[:100]",
        data_dir=data_dir,
    )
    test_set = tfds.load(
        name=name,
        split="train[:10%]",
        data_dir=data_dir,
    )
    train_set = train1.concatenate(train2)

    data_ck = iter(train_set)
    data_num = 0
    while True:
        try:
            next(data_ck)
        except:
            break
        data_num += 1

    try: labels = dataset_info.features["labels"].names
    except: labels = dataset_info.features["objects"]["label"].names

    return (train_set, valid_set, test_set), labels, data_num


def export_data(sample):
    image = sample["image"]
    gt_boxes = sample["objects"]["bbox"]
    gt_labels = sample["objects"]["label"]
    try: is_diff = sample["objects"]["is_crowd"]
    except: is_diff = sample["objects"]["is_difficult"]

    return image, gt_boxes, gt_labels, is_diff


def resize_and_rescale(image, img_size):
    transform = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size[0], img_size[1]),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)
    ])
    image = transform(image)

    return image


def evaluate(gt_boxes, gt_labels, is_diff):
    not_diff = tf.logical_not(is_diff)
    gt_boxes = gt_boxes[not_diff]
    gt_labels = gt_labels[not_diff]

    return gt_boxes, gt_labels


def rand_flip_horiz(image: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple:
    if tf.random.uniform([1]) > tf.constant([0.5]):
        image = tf.image.flip_left_right(image)
        gt_boxes = tf.stack(
            [
                gt_boxes[..., 0],
                1.0 - gt_boxes[..., 3],
                gt_boxes[..., 2],
                1.0 - gt_boxes[..., 1],
            ],
            -1,
        )

    return image, gt_boxes


def preprocess(dataset, split, img_size):
    image, gt_boxes, gt_labels, is_diff = export_data(dataset)
    image = resize_and_rescale(image, img_size)
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)
    else: 
        gt_boxes, gt_labels = evaluate(gt_boxes, gt_labels, is_diff)

    return image, gt_boxes, gt_labels
