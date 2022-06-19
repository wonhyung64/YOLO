#%%
import os
from utils import (
    load_dataset,
    build_dataset,
    load_box_prior,
)


#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    data_dir = "D:/won/data/tfds"
    img_size = (416, 416)
    batch_size = 4
    name = "coco/2017"

    datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, batch_size, img_size)
    box_prior = load_box_prior(train_set, name, img_size, data_num)
    
    image, gt_boxes, gt_labels = next(train_set)


#%%
import numpy as np
import tensorflow as tf

    gt_labels = tf.expand_dims(gt_labels, axis=-1)
    gt_labels = tf.cast(gt_labels, tf.float32)
    true_boxes = tf.concat([gt_boxes, gt_labels], axis=-1)

    anchors = box_prior

    input_shape = img_size
    input_shape = np.array(input_shape)
    m = 4
    total_labels = len(labels)

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




# %%
