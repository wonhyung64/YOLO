#%%
import os
import time
import tensorflow as tf
import numpy as np

from tqdm import tqdm
import model_utils, data_utils, loss_utils, target_utils, utils, preprocessing_utils, postprocessing_utils, test_utils, utils

#%%
hyper_params = utils.get_hyper_params()

iters = hyper_params["iters"]
batch_size = hyper_params["batch_size"]
img_size = (hyper_params["img_size"], hyper_params["img_size"])
dataset_name = hyper_params["dataset_name"]
num_classes = 20

#%%
if dataset_name == "ship":
    import ship
    dataset, labels = ship.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z, w: preprocessing_utils.preprocessing_ship(x, y, z, w))
else:
    import data_utils
    dataset, labels = data_utils.fetch_dataset(dataset_name, "train", img_size)
    dataset = dataset.map(lambda x, y, z: preprocessing_utils.preprocessing(x, y, z))
    # dataset = dataset.map(lambda x, y, z: target_utils.calculate_target(x, y, z))

data_shapes = ([None, None, None], [None, None], [None])
padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
dataset = dataset.repeat().padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values, drop_remainder=True)

dataset = iter(dataset)
img, gt_boxes, gt_labels = next(dataset)
true = generate_target(gt_boxes, gt_labels)

#%%
def generate_target(gt_boxes, gt_labels):

    gt_labels = tf.expand_dims(gt_labels, axis=-1)
    gt_labels = tf.cast(gt_labels, tf.float32)
    true_boxes = tf.concat([gt_boxes, gt_labels], axis=-1)

    anchors = utils.get_box_prior()
    hyper_params = utils.get_hyper_params()

    input_shape = [hyper_params["img_size"], hyper_params["img_size"]]
    input_shape = np.array(input_shape)
    m = hyper_params["batch_size"]

    num_layers = len(anchors) // 3

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] 

    true_boxes = np.array(true_boxes, dtype="float32")
    boxes_yx = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_hw = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_yx
    true_boxes[..., 2:4] = boxes_hw

    boxes_yx, boxes_hw = boxes_yx * input_shape, boxes_hw * input_shape

    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes), dtype="float32") for l in  range(num_layers)]

    anchors = np.expand_dims(anchors, 0)

    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_hw[..., 0] > 0

    for b in range(m):
        hw = boxes_hw[b, valid_mask[b]]
        if len(hw) == 0: continue

        hw = np.expand_dims(hw, -2)
        box_maxes = hw / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_hw = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
        box_area = hw[..., 0] * hw[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    y = np.floor(true_boxes[b,t,0] * grid_shapes[l][0]).astype("int32")
                    x = np.floor(true_boxes[b,t,1] * grid_shapes[l][1]).astype("int32") 
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype("int32")
                
                    y_true[l][b, y, x, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, y, x, k, 4] = 1.
                    y_true[l][b, y, x, k, 5+c] = 1.

    return y_true



# %%
import model_utils

input_shape = (416, 416, 3)
yolo_model = model_utils.yolo_v3(input_shape, hyper_params)

head = yolo_model(img)

#%%
#%%
def yolo_loss(yolo_outputs, true):
    anchors = utils.get_box_prior()
    num_layers = len(anchors)//3

    hyper_params = utils.get_hyper_params()
    num_classes = hyper_params["total_labels"]

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)

    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], dtype=tf.float32) for l in range(num_layers)]

    total_box_loss = tf.constant(0.)
    total_conf_loss = tf.constant(0.)
    total_cls_loss = tf.constant(0.)

    m = tf.shape(yolo_outputs[0])[0]
    mf = tf.cast(m, tf.float32)

    for l in range(num_layers):
        object_mask = true[l][..., 4:5]
        true_class_probs = true[l][..., 5:]

        grid, raw_pred, pred_yx, pred_hw, pred_obj, pred_cls = model_utils.yolo_head(yolo_outputs[l], tf.gather(anchors, anchor_mask[l]), num_classes, input_shape)

        pred_box = tf.concat([pred_yx, pred_hw], axis=-1)
    # true yxhw : 이미지 안에서 0~1
        raw_true_yx = true[l][..., :2] * grid_shapes[l][..., -1] - grid # 그리드에서 0~1
        raw_true_yx = tf.where(object_mask, raw_true_yx, tf.zeros_like(raw_true_yx)) 
        raw_true_hw = tf.math.log(true[l][..., 2:4]) / tf.gather(anchors, anchor_mask[l]) * input_shape[..., -1] # ?
        raw_true_hw = tf.where(object_mask, raw_true_hw, tf.zeros_like(raw_true_hw))
        
        box_loss_scale = 2 - true[l][..., 2:3] * true[l][..., 3:4]

        ignore_mask = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, "bool")

        def loop_cond(b, ignore_mask):
            return tf.less(b, tf.cast(m, tf.int32))
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = tf.reduce_max(iou, axis=-1)

            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < 0.5, dtype=tf.float32))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * loss_utils.bce_fn(tf.sigmoid(raw_pred[...,:2]), raw_true_yx)
        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_hw - raw_pred[...,2:4])
        wh_loss = tf.reduce_sum(wh_loss) / mf
        box_loss = xy_loss + wh_loss
        total_box_loss += box_loss

        obj_loss = object_mask * loss_utils.bce_fn(tf.sigmoid(raw_pred[..., 4:5]), object_mask) 
        noobj_loss = (1 - object_mask) * loss_utils.bce_fn(tf.sigmoid(raw_pred[..., 4:5]), object_mask) * ignore_mask
        conf_loss = tf.reduce_sum(obj_loss + noobj_loss) / mf
        total_conf_loss += conf_loss

        cls_loss = object_mask * loss_utils.bce_fn(tf.sigmoid(raw_pred[..., 5:]), true_class_probs)
        cls_loss = tf.reduce_sum(cls_loss) / mf
        total_cls_loss += cls_loss

    return total_box_loss, total_conf_loss, total_cls_loss






    


#%%
def box_iou(b1, b2):
    b1 = tf.expand_dims(b1, -2)
    b1_yx = b1[..., :2]
    b1_hw = b1[..., 2:4]
    b1_hw_half = b1_hw / 2.
    b1_mins = b1_yx - b1_hw_half
    b1_maxes = b1_yx + b1_hw_half

    b2 = tf.expand_dims(b2, 0)
    b2_yx = b2[..., :2]
    b2_hw = b2[..., 2:4]
    b2_hw_half = b2_hw / 2.
    b2_mins = b2_yx - b2_hw_half
    b2_maxes = b2_yx + b2_hw_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_hw = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]
    b1_area = b1_hw[..., 0] * b1_hw[..., 1]
    b2_area = b2_hw[..., 0] * b2_hw[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou





anchors = anchors[0]
yolo_outputs[0][...,:2]
feats.shape
feats = yolo_outputs[0]
num_anchors = 3
#%%
def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3]

    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [-1, 1, 1, 1]), [1, grid_shape[0], 1, 1])

    grid = tf.concat([grid_y, grid_x], axis=-1)
    grid = tf.cast(grid, tf.float32)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_yx = (tf.nn.sigmoid(feats[...,:2]) + grid) / tf.cast(grid_shape[...,-1], tf.float32)
    box_hw = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., -1], tf.float32)

    box_obj = tf.sigmoid(feats[..., 4:5])
    box_cls = tf.sigmoid(feats[..., 5:])

    return grid, feats, box_yx, box_hw, box_obj, box_cls









# %%

img, gt_boxes, gt_labels = next(dataset)
yolo_outputs = yolo_model(img)
true = generate_target(gt_boxes, gt_labels)
box_loss, conf_loss, cls_loss = yolo_loss(yolo_outputs, true)
print(box_loss, conf_loss, cls_loss)