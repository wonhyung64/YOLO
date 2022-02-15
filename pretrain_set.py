
# %%
import os
import numpy as np
import tensorflow as tf

import data_utils
#%%
def serialize_example(example):
    image = example["image"]
    
    image = np.array(image).tobytes()

    label = example['label']
    label = np.array(label).tobytes()
    feature_dict={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict)) 

    return example.SerializeToString()
#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 

    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    image = tf.io.decode_raw(example["image"], tf.float32)
    label = tf.io.decode_raw(example["label"], tf.int32) 

    image = tf.reshape(image, (416, 416, 3))
    return image, label

#%%
def fetch_pretrain_set(dataset, split, img_size, data_dir="D:/won", save_dir="D:/won"):

    save_dir = save_dir + "/data/crop" + dataset + "_tfrecord_" + str(img_size[0]) + "_" + str(img_size[1])
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

        # data_dir = data_dir + "/data/" + dataset + "_tfrecord_" + str(img_size[0]) + "_" + str(img_size[1])

        dataset, labels = data_utils.fetch_dataset(dataset, split, img_size, save_dir = data_dir)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = iter(dataset)
        try_num = 0
        writer = tf.io.TFRecordWriter(f"{save_dir}/{split}.tfrecord".encode("utf-8"))
        
        while True:
            img, gt_boxes, gt_labels = next(dataset)
            boxes = tf.squeeze(gt_boxes, axis=0)
            m = boxes.shape[0]
            box_indices = tf.zeros(shape=(m), dtype=tf.int32)
            crop_img = tf.image.crop_and_resize(img, boxes, box_indices, img_size)
            gt_labels = tf.squeeze(gt_labels, 0)
            gt_labels = tf.cast(gt_labels, tf.int32)
            
            for i in range(m):
                example = {
                    "image": crop_img[i],
                    "label" : gt_labels[i]
                }
                x = serialize_example(example)
                writer.write(x)
                try_num += 1
                print(try_num)

    datasets = tf.data.TFRecordDataset(f"{save_dir}/{split}.tfrecord".encode("utf-8")).map(deserialize_example)
    return datasets

#%%
dataset = fetch_pretrain_set("voc07", "train", (416,416))