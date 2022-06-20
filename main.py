#%%
import os
from utils import (
    load_dataset,
    build_dataset,
    load_box_prior,
    build_anchor_ops,
)

from tqdm import tqdm
import tensorflow as tf
def build_pos_target(anchors, gt_boxes, gt_labels, labels):
    iou_map = generate_iou(anchors, gt_boxes)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_gt_boxes = gt_boxes[valid_indices_cond]
    valid_gt_labels = gt_labels[valid_indices_cond] + 1
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)

    pos_mask = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=tf.fill(tf.shape(valid_indices)[0], True),
        shape=tf.shape(iou_map)[:2],
    )

    pos_obj = tf.where(
        pos_mask,
        tf.ones_like(pos_mask, dtype=tf.float32),
        tf.zeros_like(pos_mask, dtype=tf.float32),
    )
    pos_obj = tf.expand_dims(pos_obj, axis=-1)

    pos_reg = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=valid_gt_boxes,
        shape=tf.concat([tf.shape(iou_map)[:2], tf.constant([4], tf.int32)], axis=0),
    )

    pos_cls = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=valid_gt_labels,
        shape=tf.shape(iou_map)[:2],
    ) - 1
    pos_cls = tf.one_hot(pos_cls, len(labels), dtype=tf.float32)

    return pos_obj, pos_reg, pos_cls


def generate_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    calculate Intersection over Union

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to calculate IoU

    Returns:
        tf.Tensor: Intersection over Union
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(
        y_bottom - y_top, 0
    )

    union_area = (
        tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area
    )

    return intersection_area / union_area


#%%
if __name__ == "__main__":
    os.makedirs("data_chkr", exist_ok=True)
    data_dir = "D:/won/data/tfds"
    img_size = (416, 416)
    batch_size = 4
    name = "coco/2017"

    datasets, labels, data_num = load_dataset(name=name, data_dir=data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, batch_size, img_size)
    box_priors = load_box_prior(train_set, name, img_size, data_num)
    anchors, anchor_grids, anchor_shapes = build_anchor_ops(img_size, box_priors)
    

    for _ in tqdm(range(data_num)):
        image, gt_boxes, gt_labels = next(train_set)
        pos_obj, pos_reg, pos_cls = build_pos_target(anchors, gt_boxes, gt_labels, labels)
        
