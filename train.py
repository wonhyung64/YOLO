#%%
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import model, data, loss, target, utils

hyper_params = utils.get_hyper_params()
hyper_params["batch_size"] = 16
#%%
box_prior = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90],  [156,198],  [373,326]], dtype = np.float32)
anchors1 = box_prior[6:9] # 13,13
anchors2 = box_prior[3:6] # 26, 26
anchors3 = box_prior[0:3] # 52, 52
anchors = [anchors3, anchors2, anchors1]

#%%
yolo = model.yolo_v3((416, 416, 3), hyper_params, anchors)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
#%%
def generate_ignore_mask(true, pred):
    pred_boxes = pred[...,0:4]
    object_mask = true[..., 4:5]

    batch_size = pred_boxes.shape[0]
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_cond(idx, ignore_mask):
        return tf.less(idx, tf.cast(batch_size, tf.int32))
    def loop_body(idx, ignore_mask):
        valid_true_boxes = tf.boolean_mask(true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], "bool"))

        iou = box_iou(pred_boxes[idx], valid_true_boxes)

        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx+1, ignore_mask

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    # ignore_mask = tf.expand_dims(ignore_mask, -1)

    return ignore_mask
    
#%%
def box_iou(pred_boxes, valid_true_boxes):
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)
    
    true_box_xy = valid_true_boxes[..., 0:2]
    true_box_wh = valid_true_boxes[..., 2:4]

    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    intersection = intersect_wh[...,0] * intersect_wh[...,1]

    pred_box_area = pred_box_wh[...,0] * pred_box_wh[...,1]
    
    true_box_area = true_box_wh[...,0] * true_box_wh[...,1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    union = pred_box_area + true_box_area - intersection

    return intersection / union
#%%
# @tf.function
def train_step(images, gt_boxes, gt_labels, box_prior, hyper_params):
    with tf.GradientTape(persistent=True) as tape:
        pred = yolo(images)
        true = target.generate_target([gt_boxes, gt_labels], box_prior, hyper_params)
        ignore_mask = generate_ignore_mask(true, pred)
        coord_tune = hyper_params["coord_tune"]
        noobj_tune = hyper_params["noobj_tune"]

        box_loss = loss.box_loss(pred, true, coord_tune)
        obj_loss = loss.obj_loss(pred, true)
        nobj_loss = loss.nobj_loss(pred, true, ignore_mask, noobj_tune)
        cls_loss = loss.cls_loss(pred, true)
        total_loss = box_loss + obj_loss + nobj_loss + cls_loss
    grads = tape.gradient(total_loss, yolo.trainable_weights)
    optimizer.apply_gradients(zip(grads, yolo.trainable_weights))
    return box_loss, obj_loss, nobj_loss, cls_loss, total_loss

# %%
name = "train_10000"
data_dir = r"D:\won\coco_tfrecord"
sample = tf.data.TFRecordDataset(f"{data_dir}/{name}.tfrecord".encode("utf-8")).map(data.deserialize_example)
sample = sample.shuffle(buffer_size=10000, reshuffle_each_iteration=False)
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