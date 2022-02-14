#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import model_utils

#%%
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
imagenet_labels = np.array(open(labels_path).read().splitlines())

data_dir = "datasets/imagenet/"
write_dir = "psando/tf-imagenet-dirs"

download_config = tfds.download.DownloadConfig(
    extract_dir = os.path.join(write_dir, "extracted"),
    manual_dir=data_dir
)

download_and_prepare_kwargs = {
    "download_config" : os.path.join(write_dir, "downloaded"),
    "download_config" : download_config,
}


ds = tfds.load("imagenet2012",
               data_dir=os.path.join(write_dir, "data"),
               split="train",
               shuffle_files=False,
               download=True,
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs
               )
            
ds = tfds.load("imagenet2012",
               data_dir=os.path.join(write_dir, "data"),
               split="train",
               shuffle_files=False,
               download=True,
               as_supervised=True,
               download_and_prepare_kwargs=download_and_prepare_kwargs
               )
# %%
def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = tf.image.resize_with_crop_or_pad(i, 416, 416)
    i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
    return(i, label)
#%%
ds = ds.map(resize_with_crop)

model = model_utils.DarkNet53(include_top=True, input_shape=(416, 416, 3))

learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    power=0.4
)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"]
              )

model.fit(ds, epochs=160)

