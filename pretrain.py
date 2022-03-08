#%%
# %load_ext tensorboard
#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import model_utils

import datetime

import numpy as np

#%%
try_num = int(input("input try num"))
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
imagenet_labels = np.array(open(labels_path).read().splitlines())
imagenet_labels = np.delete(imagenet_labels, 0)
            
ds = tfds.load("imagenet2012",
               data_dir="D:/won/data/tfds",
               split="train",
               shuffle_files=False,
               download=True,
               as_supervised=True,
               )

ds_val = tfds.load("imagenet2012",
               data_dir="D:/won/data/tfds",
               split="validation",
               shuffle_files=False,
               download=True,
               as_supervised=True,
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
ds = ds.batch(32)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.map(resize_with_crop)
ds_val = ds_val.batch(32)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

model = model_utils.DarkNet53(include_top=True, input_shape=(416, 416, 3))

weights_dir = f"{os.getcwd()}/darknet_weights"
if os.listdir(weights_dir) != []:
    load_dir = f"{weights_dir}/{os.listdir(weights_dir)[-1]}/weights"
    model.load_weights(load_dir)

    save_num = int(os.listdir(weights_dir)[-1][-2:])+1
    save_dir = f"{weights_dir}/epoch{str(save_num).zfill(2)}"
    os.mkdir(save_dir)

else : 
    save_dir = f"{weights_dir}/epoch01"
    os.mkdir(save_dir)



#%%
# boundaries = list(np.array([1, 2, 3, 4, 5, 6, 7]) * 40037)
# values = list(1e-1 - (np.array([0, 1, 2, 3, 4, 5, 6, 7]) + 8*(try_num-1)) * (0.000625))
# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)



# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
# learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=0.1,
#     decay_steps=10000,
#     power=0.4
# )

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"]
              )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(ds, 
          epochs=4,
          validation_data=ds_val,
          callbacks=[tensorboard_callback]
          )

model.save_weights(f"{save_dir}/weights")


# %tensorboard --logdir logs/fit
# %%