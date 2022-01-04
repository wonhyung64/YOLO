#%%
import tensorflow as tf
import model, data
#%%

name = "validation"
data_dir = r"C:\won\data\coco"

sample = tf.data.TFRecordDataset(f"{data_dir}/{name}.tfrecord".encode("utf-8")).map(data.deserialize_example)
sample = sample.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int64))
data_shapes = ([None, None, None], [None, None], [None,])
dataset = sample.padded_batch(4, data_shapes, padding_values, drop_remainder=True)

for data in dataset.take(1):
    img, gt_boxes, gt_labels = data
    break

yolo = model.yolo_v3()
yolo.summary()
yolo(img)