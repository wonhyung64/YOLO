#%%
import numpy as np
import tensorflow as tf

import bbox_utils, utils

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

        iou = bbox_utils.bbox_iou(pred_boxes[idx], valid_true_boxes)

        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx+1, ignore_mask

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    # ignore_mask = tf.expand_dims(ignore_mask, -1)

    return ignore_mask

#%%
@tf.function
def calculate_target(img, gt_boxes, gt_labels):
    box_prior = utils.get_box_prior()
    hyper_params = utils.get_hyper_params()

    img_size = hyper_params["img_size"]
    total_labels = hyper_params["total_labels"]

    gt_boxes = tf.cast(gt_boxes * img_size, dtype=tf.float32)

    gt_ctr = tf.math.divide((gt_boxes[...,:2] + gt_boxes[...,2:]), tf.constant([2.], dtype=tf.float32)) # y x
    gt_size = gt_boxes[...,2:] - gt_boxes[...,:2] # h w
    gt_size = tf.expand_dims(gt_size, 1)
    gt_labels_onehot = tf.one_hot(gt_labels, total_labels, dtype=tf.float32)

    mins = tf.maximum(-gt_size / 2, -box_prior / 2)
    maxs = tf.minimum(gt_size / 2, box_prior / 2)
    whs = maxs - mins
    intersection = whs[...,0] * whs[...,1]
    union = gt_size[...,0] * gt_size[...,1]  + box_prior[...,0] * box_prior[...,1] - intersection
    iou_map = intersection / union
    best_iou_idx = tf.cast(tf.argmax(iou_map, axis=1), dtype=tf.int32) # box_prior idx(0~8)

    ratio_idx = tf.cast(tf.math.ceil((best_iou_idx + 1) / 3) - 1, dtype=tf.int32)
    ratio_dict= tf.constant([8., 16., 32.], dtype=tf.float32)
    ratio = tf.expand_dims(tf.gather(ratio_dict, ratio_idx), axis=-1)
    grid_idx = tf.cast(tf.multiply(gt_ctr, 1./ratio), tf.int32)

    feature_map_group = tf.cast(2 - best_iou_idx // 3, dtype=tf.int32)
    anchors_mask = tf.constant([[6,7,8], [3,4,5], [0,1,2]], dtype=tf.int32)
    box_idx = tf.argmin(tf.square(tf.gather(anchors_mask, feature_map_group) - tf.expand_dims(best_iou_idx, -1)), axis=-1, output_type=tf.int32)
    box_idx = tf.expand_dims(box_idx, axis=-1)

    gt_size = tf.squeeze(gt_size, axis=1)
    feature_map_group = tf.expand_dims(feature_map_group, axis=-1)

    gt_obj = tf.expand_dims(tf.ones_like(gt_labels, dtype=tf.float32), axis=-1)

    gt_tensor = tf.concat([
        gt_ctr, gt_size, gt_obj, gt_labels_onehot
        ], axis=-1)

    gt_idx = tf.concat([
        grid_idx, box_idx
    ], axis=-1)

    #feature_group 0
    value_tmp = tf.where(feature_map_group==0, gt_tensor, 0.)
    idx_tmp = tf.where(feature_map_group==0, gt_idx, 0)
    y_true_13 = tf.scatter_nd(idx_tmp, value_tmp, (13, 13, 3, total_labels + 5))

    #feature_group 1
    value_tmp = tf.where(feature_map_group==1, gt_tensor, 0.)
    idx_tmp = tf.where(feature_map_group==1, gt_idx, 0)
    y_true_26 = tf.scatter_nd(idx_tmp, value_tmp, (26, 26, 3, total_labels + 5))

    #feature_group 2
    value_tmp = tf.where(feature_map_group==2, gt_tensor, 0.)
    idx_tmp = tf.where(feature_map_group==2, gt_idx, 0)
    y_true_52 = tf.scatter_nd(idx_tmp, value_tmp, (52, 52, 3, total_labels + 5))

    y_true = tf.concat([
                tf.reshape(y_true_13, (y_true_13.shape[0] * y_true_13.shape[1] * 3, total_labels + 5)),
                tf.reshape(y_true_26, (y_true_26.shape[0] * y_true_26.shape[1] * 3, total_labels + 5)),
                tf.reshape(y_true_52, (y_true_52.shape[0] * y_true_52.shape[1] * 3, total_labels + 5))
                ], axis=0)
            
    return img, y_true

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
    total_labels = hyper_params["total_labels"]

    num_layers = len(anchors) // 3

    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] 

    true_boxes = np.array(true_boxes, dtype="float32")
    boxes_yx = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_hw = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_yx
    true_boxes[..., 2:4] = boxes_hw

    boxes_yx, boxes_hw = boxes_yx * input_shape, boxes_hw * input_shape

    grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]

    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+total_labels), dtype="float32") for l in  range(num_layers)]

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
