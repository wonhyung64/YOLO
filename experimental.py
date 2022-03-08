#%%
import os
import time
import tensorflow as tf

from tqdm import tqdm

import model_utils, data_utils, utils, preprocessing_utils, postprocessing_utils, test_utils, pretrain_set

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


#%%
hyper_params = utils.get_hyper_params()

hyper_params["batch_size"] = batch_size = 1
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]

dataset = pretrain_set.fetch_pretrain_set(dataset_name, "train", (416, 416), save_dir="/home1/wonhyung64")

dataset = dataset.repeat().batch(4)
dataset = dataset.shuffle(buffer_size=8000, reshuffle_each_iteration=True)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = iter(dataset)


#%%
# weights_dir = os.getcwd() + "/yolo_atmp"
# weights_dir = weights_dir + "/" + os.listdir(weights_dir)[1]

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)
# yolo_model.summary()
# yolo_model.load_weights(weights_dir + '/yolo_weights/weights')

x = yolo_model.get_layer("add_22").output
x = GlobalAveragePooling2D(name="avgpooling")(x)
output_x = Dense(20, activation="softmax", name="fc")(x)

darknet = Model(inputs=yolo_model.input, outputs=output_x)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

@tf.function
def train_step(img, true):
    with tf.GradientTape() as tape:
        pred = darknet(img)
        loss = -tf.math.reduce_sum(true * tf.math.log(tf.clip_by_value(pred,1e-9, 1))) / pred.shape[0]

    grads = tape.gradient(loss, darknet.trainable_weights)
    optimizer.apply_gradients(zip(grads, darknet.trainable_weights))

    return loss


#%%
crop_size = (416, 416)
hyper_params["iters"]=320000

step = tf.Variable(0, trainable=False)

progress_bar = tqdm(range(hyper_params['iters']))

progress_bar.set_description('iterations {}/{} | current loss ?'.format(step.numpy(), hyper_params['iters']))

start_time = time.time()

for _ in progress_bar:
    img, label = next(dataset)
    true = tf.one_hot(tf.squeeze(label, axis=-1), 20)
    loss = train_step(img, true)

    step.assign_add(1)

    progress_bar.set_description('iterations {}/{} | loss {:.4f}'.format(
        step.numpy(), hyper_params['iters'], loss.numpy()
    )) 

    if step % 500 == 0 : 
        print(progress_bar.set_description('iterations {}/{} | loss {:.4f}'.format(
            step.numpy(), hyper_params['iters'], loss.numpy()
        )) )
    
    if step % 10000 == 0 : 
        darknet.save_weights("/home1/wonhyung64/wd/yolo_atmp/darknet/weights")
        print("Weights Saved")

print("Time taken: %.2fs" % (time.time() - start_time))
# %%
