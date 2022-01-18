#%%
import tensorflow as tf
import numpy as np
import model, data, utils

#%%
hyper_params = utils.get_hyper_params()
hyper_params["batch_size"] = 1

box_prior = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

data_dir = r"D:\won\coco_tfrecord"

#%%
yolo = model.yolo_v3((416, 416, 3), hyper_params, anchors)
yolo.load_weights(data_dir + r'\yolo_weights\weights')

# %%
name = "train_10000"
sample = tf.data.TFRecordDataset(f"{data_dir}/{name}.tfrecord".encode("utf-8")).map(data.deserialize_example)
# sample = sample.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int64))
data_shapes = ([None, None, None], [None, None], [None,])
sample = sample.repeat().padded_batch(hyper_params["batch_size"], data_shapes, padding_values, drop_remainder=True)
dataset = sample.prefetch(tf.data.experimental.AUTOTUNE)
data_generator = iter(dataset)
# %%
for i in range(30):
    images, gt_boxes, gt_labels = next(data_generator)
    # images = tf.image.resize(images, (416,416))
    pred = yolo(images)
    box, score, label = utils.nms(pred, hyper_params)
    utils.draw(images, box, score, label, hyper_params)
# %%
