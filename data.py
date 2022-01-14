# %%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
def download_dataset(dataset_name, data_dir):

    if dataset_name == "coco17":
        dataset = tfds.load(name="coco/2017", data_dir=data_dir)
        train = dataset["train"]
        validation = dataset["validation"]
        test = dataset["test"]
        
    elif dataset_name == "voc07":
        dataset = tfds.load(name="voc/2007", data_dir=data_dir)
        train = dataset["train"]
        validation = dataset["validation"]
        test = dataset["test"]


    elif dataset_name == "voc12":
        dataset = tfds.load(name="voc/2012", data_dir=data_dir)
        train = dataset["train"]
        validation = dataset["validation"]
        test = dataset["test"]

    return train, validation, test
#%%
def serialize_example(example):
    image = example["image"]
    image = tf.image.resize(image, (416, 416))
    image_shape = image.shape
    
    image = np.array(image).tobytes()
    image_shape = np.array(image_shape).tobytes()

    bbox = example["bbox"]
    bbox_shape = bbox.shape

    bbox = np.array(bbox).tobytes()
    bbox_shape = np.array(bbox_shape).tobytes()

    label = example['label']
    label = np.array(label).tobytes()
    feature_dict={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        'bbox_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)) 

    return example.SerializeToString()
#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'image_shape': tf.io.FixedLenFeature([], tf.string), 
        'bbox': tf.io.FixedLenFeature([], tf.string),
        'bbox_shape': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 

    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    label = tf.io.decode_raw(example["label"], tf.int64) 

    image = tf.reshape(image,image_shape)
    bbox = tf.reshape(bbox, bbox_shape)
    return image, bbox, label

#%%