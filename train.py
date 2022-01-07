#%%
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import model, data, loss, target

#%%
box_prior = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

hyper_params = {
    "batch_size" : 16,
    "img_size" : 416,
    "total_class" : 80,
    "iterations" : 10000,
}

#%%
yolo = model.yolo_v3((416, 416, 3), hyper_params, anchors)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
#%%
# @tf.function
def train_step(images, gt_boxes, gt_labels, box_prior, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        pred = yolo(images)
        true = target.generate_target([gt_boxes, gt_labels], box_prior, hyper_params)

        box_loss = loss.box_loss(pred, true)
        obj_loss = loss.obj_loss(pred, true)
        nobj_loss = loss.obj_loss(pred, true)
        cls_loss = loss.cls_loss(pred, true)
        total_loss = box_loss + obj_loss + nobj_loss + cls_loss
    grads = tape.gradient(total_loss, yolo.trainable_weights)
    optimizer.apply_gradients(zip(grads, yolo.trainable_weights))
    return box_loss, obj_loss, nobj_loss, cls_loss, total_loss

# %%
name = "validation"
data_dir = r"C:\won\data\coco"
sample = tf.data.TFRecordDataset(f"{data_dir}/{name}.tfrecord".encode("utf-8")).map(data.deserialize_example)
# sample = sample.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int64))
data_shapes = ([None, None, None], [None, None], [None,])
sample = sample.repeat().padded_batch(hyper_params["batch_size"], data_shapes, padding_values, drop_remainder=True)
dataset = sample.prefetch(tf.data.experimental.AUTOTUNE)
data_generator = iter(dataset)
# %%
step = 0
progress_bar = tqdm(range(hyper_params['iterations']))
progress_bar.set_description('Iterations {}/{} | current loss ?'.format(step, hyper_params['iterations']))

start_time = time.time()
for _ in progress_bar:
    images, gt_boxes, gt_labels = next(data_generator)
    box_loss, obj_loss, nobj_loss, cls_loss, total_loss = train_step(images, gt_boxes, gt_labels, box_prior, hyper_params)

    step += 1
    progress_bar.set_description('Iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
        step, hyper_params['iterations'], 
        box_loss.numpy(), obj_loss.numpy(), nobj_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
    )) 
    
    if step % 1000 == 0 : 
        print(progress_bar.set_description('Iterations {}/{} | box_loss {:.4f}, obj_loss {:.4f}, nobj_loss {:.4f}, cls_loss {:.4f}, loss {:.4f}'.format(
            step, hyper_params['iterations'], 
            box_loss.numpy(), obj_loss.numpy(), nobj_loss.numpy(), cls_loss.numpy(), total_loss.numpy()
        )))
        yolo.save_weights(data_dir + r'\yolo_weights\weights')

print("Time taken: %.2fs" % (time.time() - start_time))
# %%